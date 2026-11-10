"""JSON-lines PaddleOCR service, launched only by the isolated OCR interpreter."""
import base64
from contextlib import redirect_stdout
import json
import os
import sys

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("OMP_NUM_THREADS", "2")


def main():
    engine = None
    for line in sys.stdin:
        try:
            with redirect_stdout(sys.stderr):
                import cv2
                import numpy as np
                if engine is None:
                    from paddleocr import PaddleOCR
                    engine = PaddleOCR(lang="en", device="cpu", cpu_threads=2,
                                       text_detection_model_name="PP-OCRv5_mobile_det",
                                       text_recognition_model_name="en_PP-OCRv5_mobile_rec",
                                       use_doc_orientation_classify=False, use_doc_unwarping=False,
                                       use_textline_orientation=True, enable_mkldnn=False)
                request = json.loads(line)
                image = cv2.imdecode(np.frombuffer(base64.b64decode(request['image']), np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Invalid crop image")
                rows = []
                for result in engine.predict(image):
                    rows.extend((str(text), float(score)) for text, score in
                                zip(result['rec_texts'], result['rec_scores']) if str(text).strip())
                # Never drop a low-confidence line and confirm an incomplete number.
                response = dict(text=" ".join(text for text, _ in rows),
                                confidence=min((score for _, score in rows), default=0.))
        except Exception as exc:
            response = dict(error=f"{type(exc).__name__}: {exc}")
        print('PACKET_OCR:'+json.dumps(response), flush=True)


if __name__ == '__main__':
    main()
