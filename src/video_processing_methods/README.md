# Get a thermal video with a consistent FPS

## First step: Record frames with src/video_processing_methods/capture_lepton_timestamps
e.g. 
s. description in file

to see what is recorded while the recording:
after file above is started, start live_thermal_viewer.py

## Second step: Interpolate if needed to get consistent FPS
src/video_processing_methods/get_consistent_fps_for_gray_images.py

## Third step:
Create video with src/video_processing_methods/thermal_video_generator.py

The video is used to get the ROIs but the .gray files are used to get actual temperature

## Optional step:
Get telemetry information by running read_lepton_telemtry.c
Caution: Everything besides the dump might not be accurate in its conclusions (e.g. whether FFC activated or not - interpretation might not be correct)
Meaning can be seen and double-checked in Lepton_Engineering_Datasheet

Information from ChatGPT:
In your dump:

Your label	Lepton name	Meaning
Row 1	Row A	Status, flags, output format
Row 2	Row B	Temperatures, calibration, counters
Row 3	Row C	Gain, radiometry, TLinear, configuration

So interpretation is the same as before.

ROW 3 (your Row 3) — configuration & radiometry

This is the most important row.

Gain mode (C5)
Row 3[5] = 0000


→ High gain selected

Effective gain mode (C6)
Row 3[6] = 0000


→ High gain actually active

✅ No auto-gain switching
✅ Maximum sensitivity

TLinear enable (C48)
Row 3[48] = 0001


→ TLinear enabled

TLinear resolution (C49)
Row 3[49] = 0001


→ 0.01 K resolution

✅ Radiometry enabled
✅ Highest available thermal resolution

Radiometric stability indicators
Row 3[52–54] ≈ 7301 7309 72F5


These are internal temperature references, not errors.
Values are stable and close → thermal equilibrium.

ROW 1 (your Row 1) — status & output
Status word (A4)
Row 1[4] = 73BC


Key bits:

Bit	Meaning	Value
12	AGC enabled	0
3	FFC desired	0
4–5	FFC state	idle

✅ AGC disabled
✅ No FFC happening
✅ Radiometric data valid

Video output format (A72–A73)

From Row 1:

Row 1[72] ≈ 736B
Row 1[73] ≈ 736F


Interpreted enum:

RAW14

✅ Full 14-bit data preserved
✅ No RGB / AGC pipeline active

ROW 2 (your Row 2) — calibration & temperatures

This row contains:

FPA temperature

Housing temperature

Shutter calibration constants

Timing counters

Factory calibration values

Example:

Row 2[1] = 5236


This is an internal ADC temperature representation, not scene temperature.

Nothing here indicates:

Errors

Over-temperature

Calibration failure

✅ Normal operation

 