import cv2
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

MEDICINE_KNOWLEDGE = {
    "acetaminophen": {
        "use": "Fever & pain relief",
        "warning": "Avoid overdose (max 4g/day)"
    },
    "paracetamol": {
        "use": "Fever & pain relief",
        "warning": "Avoid overdose (max 4g/day)"
    },
    "guaifenesin": {
        "use": "Helps loosen mucus (cough expectorant)",
        "warning": "Drink plenty of water"
    },
    "dextromethorphan": {
        "use": "Suppresses dry cough",
        "warning": "May cause drowsiness"
    },
    "benzonatate": {
        "use": "Reduces cough reflex",
        "warning": "Do not chew capsules"
    }
}


def interpret_schedule(schedule_text):
    schedule_text = schedule_text.strip()

    mapping = {
        "1-0-1": "Morning and night",
        "1-1-1": "Morning, afternoon, and night",
        "1-0-0": "Morning only",
        "0-1-0": "Afternoon only",
        "0-0-1": "Night only",
        "1-1-0": "Morning and afternoon",
        "0-1-1": "Afternoon and night",
        "1/2-0-1/2": "Half tablet morning and half tablet night",
        "1-0-1-0": "Morning and night",
        "1-1-1-1": "Four times a day"
    }

    return mapping.get(schedule_text, f"Schedule noted as {schedule_text}")


def normalize_med_type(raw_type):
    if not raw_type:
        return "Medicine"

    raw_type = raw_type.lower().strip()

    type_map = {
        "tab": "Tablet",
        "tablet": "Tablet",
        "cap": "Capsule",
        "capsule": "Capsule",
        "syp": "Syrup",
        "syrup": "Syrup",
        "inj": "Injection",
        "injection": "Injection",
        "drop": "Drops",
        "drops": "Drops",
        "ointment": "Ointment",
        "cream": "Cream"
    }

    return type_map.get(raw_type, raw_type.capitalize())


def is_noise_line(line):
    low = line.lower().strip()

    noise_keywords = [
        "workplace medical center",
        "medical center",
        "springville",
        "supplies",
        "www.",
        ".com",
        "phone",
        "fax",
        "address",
        "vit",
        "prescription security features",
        "secure features",
        "tamper proof",
        "microprint",
        "consumption timings",
        "timing code"
    ]

    if any(keyword in low for keyword in noise_keywords):
        return True

    if len(low) <= 2:
        return True

    return False


def clean_line(line):
    line = re.sub(r"[^\w\s:/.,()\-]", " ", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def extract_strength(text):
    patterns = [
        r"\b\d+(?:\.\d+)?\s?(?:mg|g|mcg|ml|iu)\b",
        r"\b\d+(?:\.\d+)?/\d+(?:\.\d+)?\s?(?:mg|ml)\b"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()

    return "Not specified"


def is_valid_schedule(schedule_text):
    if not schedule_text or schedule_text == "Not specified":
        return False

    parts = schedule_text.split("-")

    if len(parts) < 3 or len(parts) > 4:
        return False

    for part in parts:
        part = part.strip()

        if "/" in part:
            subparts = part.split("/")
            if len(subparts) != 2:
                return False
            for s in subparts:
                if not s.isdigit():
                    return False
                if int(s) < 0 or int(s) > 2:
                    return False
        else:
            if not part.isdigit():
                return False
            if int(part) < 0 or int(part) > 2:
                return False

    return True


def extract_schedule(text):
    matches = re.findall(r"\b\d+(?:/\d+)?-\d+(?:/\d+)?-\d+(?:/\d+)?(?:-\d+(?:/\d+)?)?\b", text)

    for match in matches:
        if is_valid_schedule(match):
            return match.strip()

    return "Not specified"


def extract_frequency(text):
    text_low = text.lower()

    found = []

    hour_match = re.search(r"every\s+(\d+)\s+(hour|hours|hr|hrs)", text_low, re.IGNORECASE)
    if hour_match:
        num = hour_match.group(1)
        found.append(f"Every {num} hours")

    day_match = re.search(r"every\s+(\d+)\s+(day|days)", text_low, re.IGNORECASE)
    if day_match:
        num = day_match.group(1)
        found.append(f"Every {num} days")

    phrase_patterns = [
        (r"once\s+daily", "Once daily"),
        (r"twice\s+daily", "Twice daily"),
        (r"thrice\s+daily", "Thrice daily"),
        (r"once\s+a\s+day", "Once a day"),
        (r"twice\s+a\s+day", "Twice a day"),
        (r"three\s+times\s+a\s+day", "Three times a day"),
        (r"before\s+food", "Before food"),
        (r"after\s+food", "After food"),
        (r"before\s+meals", "Before meals"),
        (r"after\s+meals", "After meals"),
        (r"at\s+bedtime", "At bedtime"),
        (r"\bmorning\b", "Morning"),
        (r"\bafternoon\b", "Afternoon"),
        (r"\bnight\b", "Night"),
        (r"\bbid\b", "Twice daily"),
        (r"\btid\b", "Three times a day"),
        (r"\bqid\b", "Four times a day"),
        (r"\bq6h\b", "Every 6 hours"),
        (r"\bq8h\b", "Every 8 hours"),
        (r"\bq12h\b", "Every 12 hours")
    ]

    for pattern, label in phrase_patterns:
        if re.search(pattern, text_low, re.IGNORECASE) and label not in found:
            found.append(label)

    prn_found = "prn" in text_low

    if found and prn_found:
        return ", ".join(found) + " (PRN)"
    elif found:
        return ", ".join(found)
    elif prn_found:
        return "As needed (PRN)"
    else:
        return "Timing not clearly detected"


def clean_medicine_name(name):
    if not name:
        return "Unknown"

    name = re.sub(r"\bRx\b[:\-]?", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\b\d+(?:\.\d+)?\s?(mg|ml|g|mcg|iu)\b", "", name, flags=re.IGNORECASE)

    name = re.sub(
        r"\b(by mouth|every|hours?|prn|timing|code|consumption|after|before|food|meals|bid|tid|qid|q6h|q8h|q12h)\b",
        "",
        name,
        flags=re.IGNORECASE
    )

    name = re.sub(r"/[A-Za-z]+", "", name)
    name = re.sub(r"\b\d+\b", "", name)
    name = re.sub(r"[:\-]+", " ", name)
    name = re.sub(r"\([^)]*$", "", name)

    name = re.sub(r"\s+", " ", name).strip(" -:.,()/")

    if len(name) < 2:
        return "Unknown"

    return name


def get_medicine_insight(name):
    name = name.lower()

    for key in MEDICINE_KNOWLEDGE:
        if key in name:
            return MEDICINE_KNOWLEDGE[key]

    return {
        "use": "General medication",
        "warning": "Consult a doctor for proper usage"
    }


def detect_document_type(cleaned_lines):
    joined_text = " ".join(cleaned_lines).lower()

    discharge_keywords = [
        "discharge summary",
        "date of admission",
        "date of discharge",
        "admission date",
        "discharge date",
        "hospital course",
        "physical examination findings",
        "medications at discharge",
        "discharge medications",
        "follow up",
        "follow-up",
        "continued with home meds"
    ]

    lab_keywords = [
        "hemoglobin", "wbc", "rbc", "blood", "test", "lab",
        "platelet", "cbc", "serum", "urine"
    ]

    prescription_keywords = [
        "tablet", "tab", "capsule", "cap", "rx", "prescription",
        "dose", "syrup", "inj"
    ]

    discharge_score = sum(1 for keyword in discharge_keywords if keyword in joined_text)
    lab_score = sum(1 for keyword in lab_keywords if keyword in joined_text)
    prescription_score = sum(1 for keyword in prescription_keywords if keyword in joined_text)

    if discharge_score >= 2:
        return "Discharge Summary"
    if lab_score >= 2:
        return "Lab Report"
    if prescription_score >= 2:
        return "Prescription"

    if any(keyword in joined_text for keyword in ["discharge", "admitted", "diagnosis", "hospital"]):
        return "Discharge Summary"
    if any(keyword in joined_text for keyword in ["hemoglobin", "wbc", "rbc", "blood", "test", "lab"]):
        return "Lab Report"
    if any(keyword in joined_text for keyword in ["tablet", "tab", "capsule", "cap", "rx", "prescription", "dose", "syrup", "inj"]):
        return "Prescription"

    return "Medical Document"


def is_probable_medicine_line(line):
    low = line.lower().strip()

    if is_noise_line(low):
        return False

    if re.search(r"^(tab|tablet|cap|capsule|syp|syrup|inj|injection|drop|drops|ointment|cream)\b", low, re.IGNORECASE):
        return True

    return False


def parse_medicine_line(line):
    pattern = re.compile(
        r"(?i)^(tab|tablet|cap|capsule|syp|syrup|inj|injection|drop|drops|ointment|cream)\b[\s:.-]*(.+)$"
    )

    match = pattern.search(line)
    if not match:
        return None

    med_type = normalize_med_type(match.group(1))
    remainder = match.group(2).strip()

    strength = extract_strength(remainder)
    schedule = extract_schedule(remainder)
    timing = extract_frequency(remainder)

    name = remainder
    if strength != "Not specified":
        name = re.sub(re.escape(strength), "", name, flags=re.IGNORECASE)
    if schedule != "Not specified":
        name = re.sub(re.escape(schedule), "", name, flags=re.IGNORECASE)

    name = clean_medicine_name(name)

    final_timing = interpret_schedule(schedule) if schedule != "Not specified" else timing

    return {
        "type": med_type,
        "name": name,
        "strength": strength,
        "schedule": schedule,
        "timing": final_timing if final_timing else "Timing not clearly detected"
    }


def merge_medicine_context(base_med, context_lines):
    context_text = " ".join(context_lines).strip()

    if base_med["strength"] == "Not specified":
        strength = extract_strength(context_text)
        if strength != "Not specified":
            base_med["strength"] = strength

    if base_med["schedule"] == "Not specified":
        schedule = extract_schedule(context_text)
        if schedule != "Not specified":
            base_med["schedule"] = schedule
            base_med["timing"] = interpret_schedule(schedule)

    freq = extract_frequency(context_text)
    if base_med["schedule"] == "Not specified" and freq != "Timing not clearly detected":
        base_med["timing"] = freq

    if base_med["name"] == "Unknown":
        bracket_match = re.search(r"\(([^)]+)\)", context_text)
        if bracket_match:
            possible_name = clean_medicine_name(bracket_match.group(1))
            if possible_name != "Unknown":
                base_med["name"] = possible_name

    return base_med


def fallback_extract_medicines(cleaned_lines):
    medicines = []
    current_med = None

    med_type_words = [
        "tablet", "tab", "capsule", "cap", "syrup", "syp",
        "injection", "inj", "drops", "drop", "ointment", "cream"
    ]

    for line in cleaned_lines:
        low = line.lower()

        if is_noise_line(line):
            continue

        med_type_found = None
        for word in med_type_words:
            if re.search(rf"\b{word}\b", low):
                med_type_found = word
                break

        if med_type_found:
            if current_med:
                medicines.append(current_med)

            med_type = normalize_med_type(med_type_found)

            temp_name = re.sub(rf"(?i)\b{med_type_found}\b", "", line).strip(" -:.,")
            temp_strength = extract_strength(line)
            temp_schedule = extract_schedule(line)
            temp_timing = extract_frequency(line)

            if temp_strength != "Not specified":
                temp_name = re.sub(re.escape(temp_strength), "", temp_name, flags=re.IGNORECASE)
            if temp_schedule != "Not specified":
                temp_name = re.sub(re.escape(temp_schedule), "", temp_name, flags=re.IGNORECASE)

            current_med = {
                "type": med_type,
                "name": clean_medicine_name(temp_name),
                "strength": temp_strength,
                "schedule": temp_schedule,
                "timing": interpret_schedule(temp_schedule) if temp_schedule != "Not specified" else temp_timing
            }
            continue

        if current_med:
            strength = extract_strength(line)
            schedule = extract_schedule(line)
            frequency = extract_frequency(line)

            if current_med["strength"] == "Not specified" and strength != "Not specified":
                current_med["strength"] = strength

            if current_med["schedule"] == "Not specified" and schedule != "Not specified":
                current_med["schedule"] = schedule
                current_med["timing"] = interpret_schedule(schedule)

            if current_med["schedule"] == "Not specified" and frequency != "Timing not clearly detected":
                current_med["timing"] = frequency

    if current_med:
        medicines.append(current_med)

    return medicines


def deduplicate_medicines(medicines):
    def norm_text(value):
        return str(value).strip() if value else ""

    def norm_strength(value):
        value = norm_text(value).lower()
        value = re.sub(r"\s+", "", value)
        return value

    merged_groups = {}

    for med in medicines:
        med_type = norm_text(med.get("type", "Medicine"))
        med_name = clean_medicine_name(med.get("name", "Unknown"))
        med_strength = norm_text(med.get("strength", "Not specified"))
        med_schedule = norm_text(med.get("schedule", "Not specified"))
        med_timing = norm_text(med.get("timing", "Timing not clearly detected"))

        if med_schedule == "Not specified":
            timing_low = med_timing.lower()
            if "every 6" in timing_low or "four times" in timing_low:
                med_schedule = "1-1-1-1"
            elif "every 8" in timing_low:
                med_schedule = "1-1-1"
            elif "every 12" in timing_low:
                med_schedule = "1-0-1"

        if med_timing == "Timing not clearly detected" and med_schedule != "Not specified":
            med_timing = interpret_schedule(med_schedule)

        if med_name == "Unknown" and med_strength == "Not specified" and med_schedule == "Not specified":
            continue

        group_key = (
            med_type.lower(),
            norm_strength(med_strength)
        )

        if group_key not in merged_groups:
            merged_groups[group_key] = {
                "type": med_type,
                "name": med_name,
                "strength": med_strength,
                "schedule": med_schedule,
                "timing": med_timing
            }
        else:
            existing = merged_groups[group_key]

            if existing["name"] == "Unknown" and med_name != "Unknown":
                existing["name"] = med_name

            if existing["strength"] == "Not specified" and med_strength != "Not specified":
                existing["strength"] = med_strength

            if existing["schedule"] == "Not specified" and med_schedule != "Not specified":
                existing["schedule"] = med_schedule

            if existing["timing"] == "Timing not clearly detected" and med_timing != "Timing not clearly detected":
                existing["timing"] = med_timing

            if existing["timing"] == "Timing not clearly detected" and existing["schedule"] != "Not specified":
                existing["timing"] = interpret_schedule(existing["schedule"])

    final_medicines = list(merged_groups.values())

    cleaned_final = []
    for med in final_medicines:
        if med["name"] == "Unknown":
            better_exists = any(
                other is not med
                and other["type"].lower() == med["type"].lower()
                and norm_strength(other["strength"]) == norm_strength(med["strength"])
                and other["name"] != "Unknown"
                for other in final_medicines
            )
            if better_exists:
                continue
        cleaned_final.append(med)

    for med in cleaned_final:
        insight = get_medicine_insight(med["name"])
        med["use"] = insight["use"]
        med["warning"] = insight["warning"]

    return cleaned_final


def extract_section_after_label(joined_text, labels, stop_labels=None, max_chars=200):
    if stop_labels is None:
        stop_labels = []

    for label in labels:
        pattern = rf"{label}\s*[:\-]?\s*(.+)"
        match = re.search(pattern, joined_text, re.IGNORECASE)

        if match:
            value = match.group(1)

            # Stop at next section keyword
            for stop in stop_labels:
                stop_match = re.search(rf"\b{stop}\b", value, re.IGNORECASE)
                if stop_match:
                    value = value[:stop_match.start()]

            # STOP at new line markers or separators
            value = value.split("|")[0]
            value = value.split(".")[0]

            value = value.strip(" ,.-")

            if len(value) > max_chars:
                value = value[:max_chars]

            return value.strip()

    return "Not found"


def extract_date_field(joined_text, labels):
    date_pattern = r"([0-3]?\d[\/\-][0-1]?\d[\/\-](?:\d{2}|\d{4})|[A-Za-z]{3,9}\s+[0-3]?\d,\s+\d{4}|[0-3]?\d\s+[A-Za-z]{3,9}\s+\d{4})"

    for label in labels:
        pattern = rf"{label}\s*[:\-]?\s*{date_pattern}"
        match = re.search(pattern, joined_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return "Not found"


def extract_follow_up_lines(cleaned_lines):
    follow_keywords = [
        "follow up", "follow-up", "review after", "review in", "advice", "instructions",
        "discharge advice", "recommendation", "recommended"
    ]

    results = []

    for idx, line in enumerate(cleaned_lines):
        low = line.lower()
        if any(keyword in low for keyword in follow_keywords):
            results.append(line)

            for j in range(idx + 1, min(idx + 4, len(cleaned_lines))):
                next_line = cleaned_lines[j]
                if len(next_line) > 3 and not any(k in next_line.lower() for k in ["diagnosis", "medication", "summary", "history"]):
                    results.append(next_line)
                else:
                    break
            break

    return results[:4]


def extract_discharge_medications(cleaned_lines):
    meds = []
    capture = False

    for line in cleaned_lines:
        low = line.lower()

        if "medications" in low:
            capture = True
            continue

        if capture:
            if any(k in low for k in ["follow", "advice", "diagnosis", "summary"]):
                break

            # Only take medicine-like lines
            if any(word in low for word in ["tablet", "capsule", "syrup", "mg"]):
                meds.append(line)

    return meds[:5]


def extract_discharge_details(cleaned_lines):
    joined_text = " | ".join(cleaned_lines)

    patient_name = extract_section_after_label(
        joined_text,
        labels=[r"patient name", r"name", r"patient"],
        stop_labels=["age", "sex", "gender", "mrn", "uhid", "admission", "diagnosis"],
        max_chars=80
    )

    admission_date = extract_date_field(
        joined_text,
        labels=[r"admission date", r"date of admission", r"admitted on"]
    )

    discharge_date = extract_date_field(
        joined_text,
        labels=[r"discharge date", r"date of discharge", r"discharged on", r"date"]
    )

    diagnosis = extract_section_after_label(
        joined_text,
        labels=[r"diagnosis", r"final diagnosis", r"provisional diagnosis", r"impression"],
        stop_labels=["treatment", "medication", "advice", "follow up", "follow-up", "discharge date", "summary"],
        max_chars=220
    )

    follow_up = extract_follow_up_lines(cleaned_lines)
    discharge_meds = extract_discharge_medications(cleaned_lines)

    return {
        "patient_name": patient_name,
        "admission_date": admission_date,
        "discharge_date": discharge_date,
        "diagnosis": diagnosis,
        "discharge_medications": discharge_meds,
        "follow_up": follow_up
    }


def build_discharge_summary_text(discharge_details):
    parts = []

    if discharge_details["patient_name"] != "Not found":
        parts.append(f"Patient: {discharge_details['patient_name']}")

    if discharge_details["admission_date"] != "Not found":
        parts.append(f"Admission: {discharge_details['admission_date']}")

    if discharge_details["discharge_date"] != "Not found":
        parts.append(f"Discharge: {discharge_details['discharge_date']}")

    if discharge_details["diagnosis"] != "Not found":
        parts.append(f"Diagnosis: {discharge_details['diagnosis']}")

    if discharge_details["discharge_medications"]:
        meds = " | ".join(discharge_details["discharge_medications"][:2])
        parts.append(f"Medications: {meds}")

    if discharge_details["follow_up"]:
        follow = " | ".join(discharge_details["follow_up"][:2])
        parts.append(f"Follow-up: {follow}")

    return " | ".join(parts)


def extract_medicines(cleaned_lines):
    medicines = []
    total_lines = len(cleaned_lines)

    for i, line in enumerate(cleaned_lines):
        if not is_probable_medicine_line(line):
            continue

        parsed = parse_medicine_line(line)
        if not parsed:
            continue

        context_lines = []
        for j in range(i + 1, min(i + 6, total_lines)):
            next_line = cleaned_lines[j]

            if is_probable_medicine_line(next_line):
                break

            if not is_noise_line(next_line):
                context_lines.append(next_line)

        parsed = merge_medicine_context(parsed, context_lines)
        medicines.append(parsed)

    if not medicines:
        medicines = fallback_extract_medicines(cleaned_lines)

    medicines = deduplicate_medicines(medicines)
    return medicines


def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.convertScaleAbs(gray, alpha=1.7, beta=10)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return thresh


def run_ocr(thresh):
    custom_config_1 = r'--oem 3 --psm 6'
    text1 = pytesseract.image_to_string(thresh, config=custom_config_1)

    custom_config_2 = r'--oem 3 --psm 11'
    text2 = pytesseract.image_to_string(thresh, config=custom_config_2)

    return text1 if len(text1) >= len(text2) else text2


def process_document(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return {
            "description": "Unable to read the uploaded document image.",
            "top_finding": "Not Applicable",
            "confidence": 0.0,
            "status": "Document Analysis",
            "top_findings": [],
            "prescription_details": [],
            "discharge_details": {}
        }

    thresh = preprocess_for_ocr(image)
    text = run_ocr(thresh)

    lines = text.split("\n")
    lines = [clean_line(line) for line in lines if clean_line(line)]

    cleaned_lines = []
    for line in lines:
        if len(line) >= 3:
            cleaned_lines.append(line)

    doc_type = detect_document_type(cleaned_lines)

    medicines = []
    discharge_details = {}
    extra_summary = ""

    if doc_type == "Prescription":
        medicines = extract_medicines(cleaned_lines)

        if medicines:
            med_lines = []
            for med in medicines[:5]:
                med_lines.append(
                    f"{med['type']} {med['name']} ({med['strength']}) - {med['timing']}"
                )
            extra_summary = " Extracted medicines: " + " | ".join(med_lines)
        else:
            extra_summary = " Medicine details were not clearly extracted."

    elif doc_type == "Discharge Summary":
        discharge_details = extract_discharge_details(cleaned_lines)
        discharge_summary_text = build_discharge_summary_text(discharge_details)

        if discharge_summary_text:
            extra_summary = " Extracted discharge details: " + discharge_summary_text
        else:
            extra_summary = " Discharge details were not clearly extracted."

    if len(cleaned_lines) > 0:
        important = cleaned_lines[:6]
        summary = (
            f"Detected document type: {doc_type}. "
            f"Document contains approximately {len(cleaned_lines)} meaningful text lines. "
            f"Key extracted content: " + " | ".join(important) + extra_summary
        )
    else:
        summary = "No meaningful text detected. Try a clearer or higher resolution image."

    return {
        "description": summary,
        "top_finding": doc_type,
        "confidence": 0.85 if len(cleaned_lines) > 0 else 0.0,
        "status": "Document Analysis",
        "top_findings": [],
        "prescription_details": medicines if doc_type == "Prescription" else [],
        "discharge_details": discharge_details if doc_type == "Discharge Summary" else {}
    }