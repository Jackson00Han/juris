# modules/compliance/v1.0.0/compliance.py
import re

def run(inputs: dict) -> dict:
    rules = inputs["rules"]
    draft = inputs["draft_text"]
    results = []
    overall = True

    for r in rules:
        rule_id = r["rule_id"]
        rtype = r["type"]
        params = r["parameters"]
        passed = True
        details = {}
        message = ""

        if rtype == "clause_presence":
            keyword = params["keyword"]
            found = keyword.lower() in draft.lower()
            passed = found
            details = { "found": found }
            message = f"Keyword '{keyword}' " + ("found" if found else "not found")
        elif rtype == "numeric_range":
            field = params["field_name"]
            minv = params.get("min")
            maxv = params.get("max")
            
            pattern = rf"{field}\s*:\s*([0-9]+)"
            m = re.search(pattern, draft)
            extracted = int(m.group(1)) if m else None
            within = True
            if extracted is not None:
                if minv is not None and extracted < minv: within = False
                if maxv is not None and extracted > maxv: within = False
            else:
                within = False
            passed = within
            details = { "field_name": field, "extracted_value": extracted, "within_range": within }
            message = f"Field '{field}' value {extracted} " + ("within" if within else "out of") + " range"
        elif rtype == "regex_check":
            pattern = params["pattern"]
            matches = list(re.finditer(pattern, draft))
            passed = len(matches) > 0
            details = { "matches": [ { "match_text": m.group(0), "start": m.start() } for m in matches ] }
            message = f"Regex '{pattern}' " + ("matched" if passed else "no matches")
        else:
            passed = False
            details = {}
            message = f"Unknown rule type: {rtype}"

        overall = overall and passed
        results.append({
            "rule_id": rule_id,
            "type": rtype,
            "passed": passed,
            "details": details,
            "message": message
        })

    return { "results": results, "overall_passed": overall }
