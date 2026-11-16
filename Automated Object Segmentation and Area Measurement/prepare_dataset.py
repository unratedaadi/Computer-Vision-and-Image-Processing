"""Prepare five source scans, manual outlines and reproducible benchmark variants."""
import json

import cv2
import numpy as np

from segment import DATA, load_image, save_image


# Hand-traced boundary vertices on a 310 x 426 preview of each source scan.
# The lists include the visible petiole. They were drawn from the scans.
POLYGONS = {
    "C_1_1_2_bot": [
        (140,86),(145,91),(151,107),(164,126),(174,145),(180,165),(192,186),
        (204,208),(217,242),(221,266),(221,290),(217,309),(208,327),(193,337),
        (174,340),(158,336),(149,329),(142,337),(132,347),(119,359),(105,371),
        (98,372),(97,366),(113,355),(125,344),(136,331),(119,331),(107,327),
        (96,318),(90,306),(86,291),(86,274),(91,250),(99,221),(109,188),
        (120,151),(130,125),(139,104),(141,93),
    ],
    "C_1_1_3_bot": [
        (129,74),(137,83),(144,94),(149,110),(152,132),(160,151),(161,176),
        (169,199),(174,216),(183,240),(185,254),(183,274),(177,288),(175,304),
        (169,321),(162,331),(153,338),(142,341),(141,358),(140,380),(139,391),
        (133,391),(133,374),(135,350),(135,341),(125,337),(117,326),(111,311),
        (107,292),(104,271),(105,248),(107,227),(107,210),(112,189),(117,172),
        (120,150),(119,137),(123,118),(126,99),
    ],
    "C_1_1_5_bot": [
        (159,92),(165,94),(169,108),(177,125),(183,141),(193,161),(197,181),
        (204,203),(209,225),(211,244),(208,262),(202,277),(192,288),(179,296),
        (166,300),(167,322),(163,326),(160,322),(159,297),(148,297),(138,293),
        (125,285),(120,274),(116,257),(116,238),(121,218),(124,201),(126,183),
        (128,168),(135,149),(139,134),(145,121),(152,109),
    ],
    "C_1_1_6_bot": [
        (125,118),(135,117),(148,124),(162,136),(176,152),(188,170),(198,186),
        (204,202),(208,219),(210,237),(210,256),(206,271),(199,286),(188,298),
        (174,304),(161,307),(161,319),(155,321),(155,303),(145,300),(133,292),
        (126,280),(120,266),(118,247),(117,227),(121,210),(122,187),(122,167),
        (123,149),(125,130),
    ],
    "C_1_1_7_bot": [
        (157,119),(161,119),(165,128),(170,141),(176,157),(183,174),(190,191),
        (196,208),(201,229),(207,248),(211,262),(212,275),(210,291),(205,304),
        (199,315),(191,322),(183,327),(172,329),(164,328),(162,324),(157,332),
        (151,340),(144,349),(137,358),(132,360),(130,357),(132,353),(139,347),
        (147,337),(154,326),(147,328),(140,326),(133,323),(125,317),(120,310),
        (116,302),(114,291),(113,279),(112,265),(114,251),(117,234),(121,219),
        (125,205),(131,188),(137,172),(142,155),(149,138),(153,126),
    ],
}


def transformed(image, gain, bias=0):
    return np.rint(np.clip(image.astype(np.float32) * gain + bias, 0, 255)).astype(np.uint8)


def composite(pairs, positions):
    canvas = np.full((420, 620, 3), (244, 246, 248), dtype=np.uint8)
    reference = np.zeros(canvas.shape[:2], dtype=np.uint8)
    for (image, mask), (x, y, width) in zip(pairs, positions):
        left, top, crop_width, crop_height = cv2.boundingRect(mask)
        crop = np.s_[top:top + crop_height, left:left + crop_width]
        height = round(crop_height * width / crop_width)
        obj = cv2.resize(image[crop], (width, height), interpolation=cv2.INTER_LINEAR)
        shape = cv2.resize(mask[crop], (width, height), interpolation=cv2.INTER_NEAREST)
        target = np.s_[y:y + height, x:x + width]
        canvas[target][shape != 0] = obj[shape != 0]
        reference[target][shape != 0] = 255
    return canvas, reference


def main():
    cases = {}
    pairs = []

    def write_case(name, image, mask, condition, count, sources):
        save_image(DATA / "inputs" / f"{name}.png", image)
        save_image(DATA / "reference" / f"{name}.png", mask)
        cases[name] = {"condition": condition, "count": count, "sources": sources}

    for number, (stem, polygon) in enumerate(POLYGONS.items(), start=1):
        source = f"{stem}.jpeg"
        original = load_image(DATA / "originals" / source)
        width = 465
        height = round(original.shape[0] * width / original.shape[1])
        image = cv2.resize(original, (width, height), interpolation=cv2.INTER_AREA)
        mask = np.zeros((height, width), dtype=np.uint8)
        points = np.rint(np.asarray(polygon) * np.array([width / 310, height / 426])).astype(np.int32)
        cv2.fillPoly(mask, [points], 255)
        pairs.append((image, mask))
        write_case(f"leaf_{number:02d}_original", image, mask, "Original lighting", 1, [source])
        write_case(f"leaf_{number:02d}_dim", transformed(image, 0.62, -12), mask, "Simulated dim light", 1, [source])
        gain = np.linspace(0.42, 1.03, width, dtype=np.float32)[None, :, None]
        write_case(f"leaf_{number:02d}_shadow", transformed(image, gain), mask, "Simulated side shadow", 1, [source])

    selected = [pairs[index] for index in (0, 2, 4)]
    sources = [f"{list(POLYGONS)[index]}.jpeg" for index in (0, 2, 4)]
    arrangements = {
        "scene_separated": ([(32, 56, 130), (255, 71, 110), (448, 57, 125)], "Separated composite"),
        "scene_touching": ([(130, 56, 130), (204, 71, 110), (273, 57, 125)], "Touching composite"),
    }
    for name, (positions, condition) in arrangements.items():
        image, mask = composite(selected, positions)
        write_case(name, image, mask, condition, len(selected), sources)

    (DATA / "cases.json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
