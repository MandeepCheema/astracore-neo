# Deep test 6 — FP32 vs fake-INT8 output drift (YOLOv8n on bus)

- Output shape:  `[1, 84, 8400]`
- SNR:           **29.7 dB**  (higher is better; > 30 dB is production-grade for INT8)
- Cosine:        **0.999469**  (1.0 = perfect alignment)
- Max abs error: 201.1427

Same raw ONNX fed through (a) plain FP32 ORT and (b) the SDK's own fake-INT8 fake-quant reference path. Low drift here is evidence that the SDK's INT8 calibration preserves the model's answer. Published YOLOv8n INT8 recipes land in the 30-45 dB range.
