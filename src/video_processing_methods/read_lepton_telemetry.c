#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
*  *   gcc -std=c11 -O2 -Wall src/video_processing_methods/read_lepton_telemetry.c -o read_lepton_telemetry 
*./read_lepton_telemetry --file /path/to/frame.gray --width 160 --height 122 --telemetry end --telemetry-rows 3 --dump-telemetry-all
*/

typedef enum { TELEMETRY_AT_START = 0, TELEMETRY_AT_END = 1 } telemetry_loc_t;

typedef struct {
    const char *file_path;
    const char *output_file;
    unsigned int width;
    unsigned int height;
    unsigned int telemetry_rows;
    telemetry_loc_t telemetry_loc;
    int frame_index; /* -1 = last */
    unsigned int status_word_index;
    unsigned int gain_word_index;
    unsigned int eff_gain_word_index;
    unsigned int tlinear_enable_word_index;
    unsigned int tlinear_res_word_index;
    unsigned int video_format_word_index;
    int dump_telemetry;
    int dump_all_telemetry_rows;
    FILE *out;
} options_t;

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s --file PATH [options]\n\n"
            "Options:\n"
            "  --width N              Frame width in pixels (default: 80)\n"
            "  --height N             Frame height including telemetry (default: 63)\n"
            "  --telemetry start|end  Telemetry placement (default: end)\n"
            "  --telemetry-rows N     Number of telemetry rows (default: 3)\n"
            "  --frame N              Frame index (0-based). Default: last frame\n"
            "  --status-word N        Row A status word index (default: 4)\n"
            "  --gain-word N          Row C gain mode word index (default: 5)\n"
            "  --eff-gain-word N      Row C effective gain word index (default: 6)\n"
            "  --tlinear-enable N     Row C TLinear enable word index (default: 48)\n"
            "  --tlinear-res N        Row C TLinear resolution word index (default: 49)\n"
            "  --video-format N       Row A video format word index (default: 72)\n"
            "  --dump-telemetry       Print all telemetry words for each telemetry row\n"
            "  --dump-telemetry-all   Print all telemetry rows (1..N) instead of A/B/C\n"
            "\nExample:\n"
            "  %s --file frame000123.gray --width 80 --height 63 --telemetry end\n",
            prog, prog);
}

static int parse_uint(const char *s, unsigned int *out) {
    char *end = NULL;
    unsigned long val = strtoul(s, &end, 10);
    if (!s[0] || (end && *end != '\0')) return -1;
    *out = (unsigned int)val;
    return 0;
}

static int parse_int(const char *s, int *out) {
    char *end = NULL;
    long val = strtol(s, &end, 10);
    if (!s[0] || (end && *end != '\0')) return -1;
    *out = (int)val;
    return 0;
}

static uint16_t read_word_le(const uint8_t *bytes, size_t word_index) {
    size_t byte_index = word_index * 2;
    uint16_t lo = bytes[byte_index];
    uint16_t hi = bytes[byte_index + 1];
    return (uint16_t)((hi << 8) | lo);
}

static const char *gain_mode_str(uint16_t v) {
    switch (v) {
        case 0: return "high";
        case 1: return "low";
        case 2: return "auto";
        default: return "unknown";
    }
}

static const char *tlinear_res_str(uint16_t v) {
    switch (v) {
        case 0: return "0.1 K";
        case 1: return "0.01 K";
        default: return "unknown";
    }
}

static const char *video_format_str(uint32_t v) {
    switch (v) {
        case 0: return "RAW8";
        case 1: return "RAW10";
        case 2: return "RAW12";
        case 3: return "RAW14";
        case 4: return "RGB888";
        default: return "unknown";
    }
}

static void dump_row_words(FILE *out, const char *label, const uint8_t *frame, size_t row_base, unsigned int width) {
    fprintf(out, "\n%s (word count: %u)\n", label, width);
    for (unsigned int i = 0; i < width; ++i) {
        uint16_t word = read_word_le(frame, row_base + i);
        if (i % 16 == 0) {
            fprintf(out, "  [%3u]", i);
        }
        fprintf(out, " %04X", word);
        if ((i + 1) % 16 == 0 || i + 1 == width) {
            fprintf(out, "\n");
        }
    }
}

int main(int argc, char **argv) {
    options_t opt = {
        .file_path = NULL,
        .output_file = NULL,
        .width = 80,
        .height = 63,
        .telemetry_rows = 3,
        .telemetry_loc = TELEMETRY_AT_END,
        .frame_index = -1,
        .status_word_index = 4,
        .gain_word_index = 5,
        .eff_gain_word_index = 6,
        .tlinear_enable_word_index = 48,
        .tlinear_res_word_index = 49,
        .video_format_word_index = 72,
        .dump_telemetry = 0,
        .dump_all_telemetry_rows = 0,
        .out = NULL,
    };

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--file") && i + 1 < argc) {
            opt.file_path = argv[++i];
        } else if (!strcmp(argv[i], "--output") && i + 1 < argc) {
            opt.output_file = argv[++i];
        } else if (!strcmp(argv[i], "--width") && i + 1 < argc) {
            if (parse_uint(argv[++i], &opt.width) != 0) {
                fprintf(stderr, "Invalid width\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--height") && i + 1 < argc) {
            if (parse_uint(argv[++i], &opt.height) != 0) {
                fprintf(stderr, "Invalid height\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--telemetry") && i + 1 < argc) {
            const char *loc = argv[++i];
            if (!strcmp(loc, "start")) opt.telemetry_loc = TELEMETRY_AT_START;
            else if (!strcmp(loc, "end")) opt.telemetry_loc = TELEMETRY_AT_END;
            else {
                fprintf(stderr, "Invalid telemetry location: %s\n", loc);
                return 2;
            }
        } else if (!strcmp(argv[i], "--telemetry-rows") && i + 1 < argc) {
            if (parse_uint(argv[++i], &opt.telemetry_rows) != 0 || opt.telemetry_rows < 1) {
                fprintf(stderr, "Invalid telemetry rows\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--frame") && i + 1 < argc) {
            if (parse_int(argv[++i], &opt.frame_index) != 0) {
                fprintf(stderr, "Invalid frame index\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--status-word") && i + 1 < argc) {
            if (parse_uint(argv[++i], &opt.status_word_index) != 0) {
                fprintf(stderr, "Invalid status-word index\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--gain-word") && i + 1 < argc) {
            if (parse_uint(argv[++i], &opt.gain_word_index) != 0) {
                fprintf(stderr, "Invalid gain-word index\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--eff-gain-word") && i + 1 < argc) {
            if (parse_uint(argv[++i], &opt.eff_gain_word_index) != 0) {
                fprintf(stderr, "Invalid eff-gain-word index\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--tlinear-enable") && i + 1 < argc) {
            if (parse_uint(argv[++i], &opt.tlinear_enable_word_index) != 0) {
                fprintf(stderr, "Invalid tlinear-enable index\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--tlinear-res") && i + 1 < argc) {
            if (parse_uint(argv[++i], &opt.tlinear_res_word_index) != 0) {
                fprintf(stderr, "Invalid tlinear-res index\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--video-format") && i + 1 < argc) {
            if (parse_uint(argv[++i], &opt.video_format_word_index) != 0) {
                fprintf(stderr, "Invalid video-format index\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--dump-telemetry")) {
            opt.dump_telemetry = 1;
        } else if (!strcmp(argv[i], "--dump-telemetry-all")) {
            opt.dump_telemetry = 1;
            opt.dump_all_telemetry_rows = 1;
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 2;
        }
    }

    if (!opt.file_path) {
        print_usage(argv[0]);
        return 2;
    }

    opt.out = stdout;
    if (opt.output_file) {
        opt.out = fopen(opt.output_file, "w");
        if (!opt.out) {
            fprintf(stderr, "Failed to open output file %s: %s\n", opt.output_file, strerror(errno));
            return 1;
        }
    }

    FILE *f = fopen(opt.file_path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s: %s\n", opt.file_path, strerror(errno));
        if (opt.output_file) fclose(opt.out);
        return 1;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "Failed to seek: %s\n", strerror(errno));
        fclose(f);
        return 1;
    }
    long size = ftell(f);
    if (size < 0) {
        fprintf(stderr, "Failed to tell file size\n");
        fclose(f);
        return 1;
    }
    rewind(f);

    size_t frame_bytes = (size_t)opt.width * (size_t)opt.height * 2;
    if ((size_t)size < frame_bytes) {
        fprintf(stderr, "File too small: %ld bytes, need %zu\n", size, frame_bytes);
        fclose(f);
        return 1;
    }

    size_t frame_count = (size_t)size / frame_bytes;
    if (frame_count == 0) {
        fprintf(stderr, "No full frames found in file\n");
        fclose(f);
        return 1;
    }

    size_t frame_index = (opt.frame_index < 0) ? (frame_count - 1) : (size_t)opt.frame_index;
    if (frame_index >= frame_count) {
        fprintf(stderr, "Frame index out of range: %zu (count=%zu)\n", frame_index, frame_count);
        fclose(f);
        return 1;
    }

    if (fseek(f, (long)(frame_index * frame_bytes), SEEK_SET) != 0) {
        fprintf(stderr, "Failed to seek to frame\n");
        fclose(f);
        return 1;
    }

    uint8_t *frame = (uint8_t *)malloc(frame_bytes);
    if (!frame) {
        fprintf(stderr, "Out of memory\n");
        fclose(f);
        return 1;
    }

    size_t read_bytes = fread(frame, 1, frame_bytes, f);
    fclose(f);
    if (read_bytes != frame_bytes) {
        fprintf(stderr, "Short read: %zu/%zu\n", read_bytes, frame_bytes);
        free(frame);
        return 1;
    }

    if (opt.telemetry_rows < 3) {
        fprintf(stderr, "Telemetry rows must be >= 3 to access Row A/B/C\n");
        free(frame);
        return 1;
    }

    int row_a = (opt.telemetry_loc == TELEMETRY_AT_END) ? (int)(opt.height - opt.telemetry_rows) : 0;
    int row_c = row_a + 2;
    if (row_a < 0 || row_c >= (int)opt.height) {
        fprintf(stderr, "Telemetry rows out of bounds (height=%u, rows=%u)\n", opt.height, opt.telemetry_rows);
        free(frame);
        return 1;
    }

    size_t row_a_base = (size_t)row_a * opt.width;
    size_t row_c_base = (size_t)row_c * opt.width;

    uint16_t gain_mode = read_word_le(frame, row_c_base + opt.gain_word_index);
    uint16_t eff_gain = read_word_le(frame, row_c_base + opt.eff_gain_word_index);
    uint16_t tlinear_enable = read_word_le(frame, row_c_base + opt.tlinear_enable_word_index);
    uint16_t tlinear_res = read_word_le(frame, row_c_base + opt.tlinear_res_word_index);
    uint16_t status_word = read_word_le(frame, row_a_base + opt.status_word_index);
    uint16_t video_format_lo = read_word_le(frame, row_a_base + opt.video_format_word_index);
    uint16_t video_format_hi = read_word_le(frame, row_a_base + opt.video_format_word_index + 1);

    uint32_t video_format = ((uint32_t)video_format_hi << 16) | video_format_lo;

    unsigned int agc_state = (status_word >> 12) & 0x1;
    unsigned int ffc_state = (status_word >> 4) & 0x3;
    unsigned int ffc_desired = (status_word >> 3) & 0x1;

    fprintf(opt.out, "Frame: %zu (total frames: %zu)\n", frame_index, frame_count);
    fprintf(opt.out, "Telemetry rows: %u (%s)\n", opt.telemetry_rows, opt.telemetry_loc == TELEMETRY_AT_END ? "end" : "start");

    fprintf(opt.out, "\nHigh gain mode\n");
    fprintf(opt.out, "  Gain Mode (C%u): %u (%s)\n", opt.gain_word_index, gain_mode, gain_mode_str(gain_mode));
    fprintf(opt.out, "  Effective Gain Mode (C%u): %u (%s)\n", opt.eff_gain_word_index, eff_gain, gain_mode_str(eff_gain));
    if (gain_mode == 0 && eff_gain == 0) {
        fprintf(opt.out, "  Status: OK (high gain active)\n");
    } else if (gain_mode == 2 && eff_gain == 0) {
        fprintf(opt.out, "  Status: OK (auto -> high gain active)\n");
    } else {
        fprintf(opt.out, "  Status: NO (not high gain)\n");
    }

    fprintf(opt.out, "\nRadiometry + TLinear\n");
    fprintf(opt.out, "  TLinear Enable (C%u): %u (%s)\n", opt.tlinear_enable_word_index, tlinear_enable, tlinear_enable ? "enabled" : "disabled");
    fprintf(opt.out, "  TLinear Resolution (C%u): %u (%s)\n", opt.tlinear_res_word_index, tlinear_res, tlinear_res_str(tlinear_res));
    if (tlinear_enable == 1) {
        fprintf(opt.out, "  Status: OK (TLinear enabled)\n");
    } else {
        fprintf(opt.out, "  Status: NO (TLinear disabled)\n");
    }

    fprintf(opt.out, "\nAGC\n");
    fprintf(opt.out, "  AGC State (Row A status bit 12): %u (%s)\n", agc_state, agc_state ? "enabled" : "disabled");
    if (agc_state == 0) {
        fprintf(opt.out, "  Status: OK (AGC disabled)\n");
    } else {
        fprintf(opt.out, "  Status: NO (AGC enabled)\n");
    }

    fprintf(opt.out, "\nVideo Output Format\n");
    fprintf(opt.out, "  A%u–A%u raw: 0x%04X 0x%04X\n", opt.video_format_word_index, opt.video_format_word_index + 1, video_format_lo, video_format_hi);
    if (video_format_hi == 0) {
        fprintf(opt.out, "  Parsed format: %u (%s)\n", video_format_lo, video_format_str(video_format_lo));
        if (video_format_lo == 3) {
            fprintf(opt.out, "  Status: OK (RAW14)\n");
        } else {
            fprintf(opt.out, "  Status: NO (not RAW14)\n");
        }
    } else {
        fprintf(opt.out, "  Parsed format: 0x%08X (unknown enum)\n", video_format);
        fprintf(opt.out, "  Status: UNKNOWN (update mapping if needed)\n");
    }

    fprintf(opt.out, "\nFFC\n");
    fprintf(opt.out, "  FFC Desired (bit 3): %u\n", ffc_desired);
    fprintf(opt.out, "  FFC State (bits 4-5): %u\n", ffc_state);
    fprintf(opt.out, "  Note: Automatic FFC is inferred by periodic FFC events without commands.\n");

    if (opt.dump_telemetry) {
        fprintf(opt.out, "\nFull telemetry dump\n");
        if (opt.dump_all_telemetry_rows) {
            for (unsigned int i = 0; i < opt.telemetry_rows; ++i) {
                char label[32];
                snprintf(label, sizeof(label), "Row %u", i + 1);
                dump_row_words(opt.out, label, frame, (size_t)(row_a + (int)i) * opt.width, opt.width);
            }
        } else {
            int row_b = row_a + 1;
            dump_row_words(opt.out, "Row A", frame, row_a_base, opt.width);
            if (row_b >= 0 && row_b < (int)opt.height) {
                dump_row_words(opt.out, "Row B", frame, (size_t)row_b * opt.width, opt.width);
            }
            dump_row_words(opt.out, "Row C", frame, row_c_base, opt.width);
        }
    }

    free(frame);
    if (opt.output_file) fclose(opt.out);
    return 0;
}
