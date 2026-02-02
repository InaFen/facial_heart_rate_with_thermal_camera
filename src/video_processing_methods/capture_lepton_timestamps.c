#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#define CLEAR(x) memset(&(x), 0, sizeof(x))

/*
 * Simple frame capture and camera-timestamp dumper for PureThermal + FLIR Lepton.
 *
 * What this program does:
 * - Opens a V4L2 video device that exposes Lepton frames (e.g. PureThermal)
 * - Captures Y16 (16-bit grayscale) frames
 * - Reads the camera's telemetry millisecond counter (time_ms) embedded in the frame
 * - Prints the timestamp for each frame and optionally saves the raw frame
 *   using either a sequential index or the camera timestamp as the filename.
 *
 * Why height can be 122 instead of 120:
 * - The image is 160x120 pixels, but when telemetry is enabled the device appends
 *   extra lines. With PureThermal, we commonly see 2 telemetry lines at the end,
 *   so the total frame height is 160x122 in Y16.
 *
 * Telemetry extraction:
 * - The camera embeds telemetry words in one of the last lines (or first lines,
 *   if telemetry-at-start). The time counter is stored as two 16-bit words in
 *   little-endian order: time_ms = (high_word << 16) | low_word.
 * - This program looks first at the second-to-last line (common on PureThermal),
 *   then the last line, and falls back to the first lines if telemetry-at-start.
 *
 * Quick usage examples:
 *   gcc -std=c11 -O2 -Wall src/video_processing_methods/capture_lepton_timestamps.c -o capture_lepton_timestamps 
 *
 *   ./capture_lepton_timestamps --device /dev/v4l/by-id/usb-GroupGets_PureThermal__fw:v1.3.0__80030034-5101-3133-3332-373300000000-video-index0 --duration 10 --telemetry end --width 160 --height 122 --save-ts /home/arbeit/Dokumente/master_thesis/data/lepton/raw_lepton_data/temp/
 */

struct buffer {
    void *start;
    size_t length;
};

/* Where telemetry lines are placed in the frame */
typedef enum { TELEMETRY_UNKNOWN = 0, TELEMETRY_AT_START, TELEMETRY_AT_END } telemetry_loc_t;

static char *dev_name = NULL;
static int fd = -1;
static struct buffer *buffers = NULL;
static unsigned int n_buffers = 0;
static unsigned int frame_limit = 100;
static double duration_sec = 0.0;  /* if > 0, capture for this many seconds instead of frame_limit frames */
static telemetry_loc_t telemetry_loc = TELEMETRY_AT_END;
static unsigned int width = 160;
static unsigned int height = 122; /* default allows room for telemetry lines */
static char *out_prefix = NULL;
static int save_frames = 0;
static char *save_ts_prefix = NULL;
static int save_by_ts = 0;

/* Print error and exit */
static void errno_exit(const char *s) {
    fprintf(stderr, "%s error %d, %s\n", s, errno, strerror(errno));
    exit(EXIT_FAILURE);
}

/* Safe wrapper around ioctl that retries on EINTR */
static int xioctl(int fh, int request, void *arg) {
    int r;
    do {
        r = ioctl(fh, request, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

/*
 * Extract the camera millisecond timestamp from the raw frame words.
 * - frame_words: pointer to the frame mapped in memory
 * - w, h: negotiated frame width/height (e.g., 160x122) -122 because telemetry lines are added
 * - loc: expected telemetry placement (start or end)
 * - out_time_ms: filled with 32-bit time in milliseconds if found
 * Returns 0 on success, 1 if telemetry wasn't found, -1 on invalid input.
 */
static int extract_time_ms_from_frame(const uint16_t *frame_words, unsigned int w, unsigned int h, telemetry_loc_t loc, uint32_t *out_time_ms) {
    if (h == 0 || w == 0) return -1;

    /*
     * Telemetry placement on PureThermal + Lepton3 in Y16 mode typically yields the
     * time counter in the second-to-last line when telemetry is appended at the end.
     * We try that first, then fall back to the last line, and finally to the first
     * line (for telemetry-at-start configurations).
     */
    const uint16_t *candidates[3];
    unsigned int candidate_count = 0;

    if (loc == TELEMETRY_AT_END || loc == TELEMETRY_UNKNOWN) {
        if (h >= 2) candidates[candidate_count++] = frame_words + (h - 2) * w;
        candidates[candidate_count++] = frame_words + (h - 1) * w;
    } else if (loc == TELEMETRY_AT_START) {
        candidates[candidate_count++] = frame_words;
        if (h >= 2) candidates[candidate_count++] = frame_words + w;
    }

    for (unsigned int i = 0; i < candidate_count; ++i) {
        const uint16_t *line = candidates[i];
        uint16_t word0 = line[0];
        uint16_t word1 = (w > 1) ? line[1] : 0;
        uint16_t word2 = (w > 2) ? line[2] : 0;

        /* Telemetry data is little-endian: low word then high word. */
        uint32_t time_ms = ((uint32_t)word2 << 16) | word1;

        /* Heuristic checks: telemetry revision is small and time_ms should change. */
        if (word0 <= 0x00FF) {
            *out_time_ms = time_ms;
            return 0;
        }
    }

    return 1; /* telemetry not detected */
}

/*
 * Called for each received frame buffer:
 * - parse telemetry and print camera_time_ms
 * - optionally save the raw frame using timestamp or a sequential index
 */
static void process_image(const void *p, int size, unsigned int frame_idx) {
    (void)size;
    uint32_t time_ms = 0;
    int status = extract_time_ms_from_frame((const uint16_t *)p, width, height, telemetry_loc, &time_ms);

    if (status == 0) {
        printf("frame %u: camera_time_ms=%u\n", frame_idx, time_ms);
    } else {
        printf("frame %u: camera_time_ms unknown (telemetry not detected)\n", frame_idx);
    }

    if (save_by_ts && save_ts_prefix && status == 0) {
        char path[256];
        /* Save using camera timestamp as filename */
        snprintf(path, sizeof(path), "%s%010u.gray", save_ts_prefix, time_ms);
        FILE *f = fopen(path, "wb");
        if (f) {
            fwrite(p, 1, (size_t)size, f);
            fclose(f);
        }
    } else if (save_frames && out_prefix) {
        char path[256];
        snprintf(path, sizeof(path), "%s%06u.gray", out_prefix, frame_idx);
        FILE *f = fopen(path, "wb");
        if (f) {
            fwrite(p, 1, (size_t)size, f);
            fclose(f);
        }
    }
}

/* Dequeue a frame from the driver, process, and requeue the buffer */
static int read_frame(unsigned int frame_idx) {
    struct v4l2_buffer buf;
    CLEAR(buf);

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (xioctl(fd, VIDIOC_DQBUF, &buf) == -1) {
        if (errno == EAGAIN) return 0;
        errno_exit("VIDIOC_DQBUF");
    }

    process_image(buffers[buf.index].start, buf.bytesused, frame_idx);

    if (xioctl(fd, VIDIOC_QBUF, &buf) == -1)
        errno_exit("VIDIOC_QBUF");

    return 1;
}

/*
 * Main capture loop:
 * - Waits for the device to become ready (select())
 * - Dequeues and processes frames until frame_limit is reached or duration expires
 */
static void mainloop(void) {
    unsigned int count = frame_limit;
    unsigned int frame_idx = 0;
    struct timeval start_time, current_time;
    int duration_mode = (duration_sec > 0.0);
    
    if (duration_mode) {
        gettimeofday(&start_time, NULL);
    }

    while (duration_mode || frame_idx < count) {
        /* Check if duration limit reached */
        if (duration_mode) {
            gettimeofday(&current_time, NULL);
            double elapsed = (current_time.tv_sec - start_time.tv_sec) + 
                            (current_time.tv_usec - start_time.tv_usec) / 1e6;
            if (elapsed >= duration_sec) {
                break;
            }
        }
        fd_set fds;
        struct timeval tv;
        int r;

        FD_ZERO(&fds);
        FD_SET(fd, &fds);

        tv.tv_sec = 2;
        tv.tv_usec = 0;

        r = select(fd + 1, &fds, NULL, NULL, &tv);

        if (r == -1) {
            if (EINTR == errno) continue;
            errno_exit("select");
        }
        if (r == 0) {
            fprintf(stderr, "select timeout\n");
            exit(EXIT_FAILURE);
        }
        if (read_frame(frame_idx)) {
            frame_idx++;
        }
    }
}

/* Stop streaming */
static void stop_capturing(void) {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd, VIDIOC_STREAMOFF, &type) == -1)
        errno_exit("VIDIOC_STREAMOFF");
}

/* Enqueue buffers and start streaming */
static void start_capturing(void) {
    unsigned int i;
    enum v4l2_buf_type type;

    for (i = 0; i < n_buffers; ++i) {
        struct v4l2_buffer buf;
        CLEAR(buf);
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (xioctl(fd, VIDIOC_QBUF, &buf) == -1)
            errno_exit("VIDIOC_QBUF");
    }

    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd, VIDIOC_STREAMON, &type) == -1)
        errno_exit("VIDIOC_STREAMON");
}

/* Unmap all buffers and free memory */
static void uninit_device(void) {
    unsigned int i;
    for (i = 0; i < n_buffers; ++i)
        munmap(buffers[i].start, buffers[i].length);
    free(buffers);
}

/* Request MMAP buffers and map them to userspace */
static void init_mmap(void) {
    struct v4l2_requestbuffers req;
    CLEAR(req);

    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (xioctl(fd, VIDIOC_REQBUFS, &req) == -1) {
        if (errno == EINVAL)
            errno_exit("device does not support memory mapping");
        else
            errno_exit("VIDIOC_REQBUFS");
    }

    if (req.count < 2)
        errno_exit("Insufficient buffer memory");

    buffers = calloc(req.count, sizeof(*buffers));
    if (!buffers)
        errno_exit("Out of memory");

    for (n_buffers = 0; n_buffers < req.count; ++n_buffers) {
        struct v4l2_buffer buf;
        CLEAR(buf);

        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = n_buffers;

        if (xioctl(fd, VIDIOC_QUERYBUF, &buf) == -1)
            errno_exit("VIDIOC_QUERYBUF");

        buffers[n_buffers].length = buf.length;
        buffers[n_buffers].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);

        if (buffers[n_buffers].start == MAP_FAILED)
            errno_exit("mmap");
    }
}

/*
 * Query device capabilities, set the pixel format and dimensions, and
 * initialize mmap buffers. Defaults to Y16 (16-bit grayscale), width=160,
 * height=122 to include telemetry lines.
 */
static void init_device(void) {
    struct v4l2_capability cap;
    struct v4l2_format fmt;

    if (xioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) {
        if (errno == EINVAL)
            errno_exit("Not a V4L2 device");
        else
            errno_exit("VIDIOC_QUERYCAP");
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE))
        errno_exit("No video capture capability");
    if (!(cap.capabilities & V4L2_CAP_STREAMING))
        errno_exit("No streaming I/O support");

    CLEAR(fmt);
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_Y16;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (xioctl(fd, VIDIOC_S_FMT, &fmt) == -1)
        errno_exit("VIDIOC_S_FMT");

    width = fmt.fmt.pix.width;
    height = fmt.fmt.pix.height;

    init_mmap();
}

/* Close the device file descriptor */
static void close_device(void) {
    if (fd != -1)
        close(fd);
}

/* Open the V4L2 device for non-blocking read/write */
static void open_device(void) {
    fd = open(dev_name, O_RDWR | O_NONBLOCK, 0);
    if (fd == -1) {
        fprintf(stderr, "Cannot open '%s': %d, %s\n", dev_name, errno, strerror(errno));
        exit(EXIT_FAILURE);
    }
}

/* Print usage information */
static void usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s --device DEV [--frames N] [--duration SEC] [--telemetry start|end] [--width W] [--height H] [--save PREFIX] [--save-ts PREFIX]\n",
            prog);
}

/*
 * Parse command line arguments:
 * --device PATH     : V4L2 device to open
 * --frames N        : number of frames to capture
 * --telemetry start|end : expected telemetry placement in frame
 * --width W --height H   : desired frame dimensions (Y16)
 * --save PREFIX     : save frames by sequential index to PREFIXNNNNNN.gray
 * --save-ts PREFIX  : save frames named by camera timestamp to PREFIX##########.gray
 */
static void parse_args(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            dev_name = argv[++i];
        } else if (!strcmp(argv[i], "--frames") && i + 1 < argc) {
            frame_limit = (unsigned int)atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--telemetry") && i + 1 < argc) {
            const char *loc = argv[++i];
            if (!strcmp(loc, "start")) telemetry_loc = TELEMETRY_AT_START;
            else if (!strcmp(loc, "end")) telemetry_loc = TELEMETRY_AT_END;
            else telemetry_loc = TELEMETRY_UNKNOWN;
        } else if (!strcmp(argv[i], "--width") && i + 1 < argc) {
            width = (unsigned int)atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--height") && i + 1 < argc) {
            height = (unsigned int)atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--save") && i + 1 < argc) {
            out_prefix = argv[++i];
            save_frames = 1;
        } else if (!strcmp(argv[i], "--save-ts") && i + 1 < argc) {
            save_ts_prefix = argv[++i];
            save_by_ts = 1;
        } else if (!strcmp(argv[i], "--duration") && i + 1 < argc) {
            duration_sec = atof(argv[++i]);
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        } else {
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    if (!dev_name) {
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv) {
    parse_args(argc, argv);

    open_device();
    init_device();
    start_capturing();

    printf("Capturing from %s (w=%u h=%u, telemetry=%s)\n", dev_name, width, height,
           telemetry_loc == TELEMETRY_AT_START ? "start" : "end");
    if (duration_sec > 0.0) {
        printf("Duration: %.1f seconds\n", duration_sec);
    } else {
        printf("Frames: %u\n", frame_limit);
    }

    mainloop();
    stop_capturing();
    uninit_device();
    close_device();
    return 0;
}
