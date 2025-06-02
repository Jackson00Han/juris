# Module Contracts

This document specifies the JSON I/O contracts for each core module in the Juris RAG API (MVP 1).  
Each module’s contract includes:

1. **Name** and **Version**  
2. **Inputs** (field name + type + description)  
3. **Outputs** (field name + type + description)  

---

## 1. Retrieval Module  
**Version:** 1.0.0  

### 1.1 Inputs  
```json
{
  "doc_id": "string",      // Unique identifier for the uploaded document’s FAISS index
  "question": "string",    // User’s natural-language query
  "top_k": "integer"       // (Optional) Number of top contexts to retrieve; default = 3
}

```
- doc_id: must match an existing FAISS index folder under faiss_indexes/.

- question: free-text string describing what the user wants to ask about the document.

- top_k: integer ≥ 1; how many text chunks to retrieve for context.

### 1.2 Outputs
```
{
  "answer": "string",      // LLM-generated answer based on retrieved contexts
  "context": "string"      // Concatenated top_k text chunks (joined by “\n\n”)
}
```
- answer: the LLM’s response to the question, extracted from its “generated_text” field.

- context: one string containing the top_k retrieved chunks, separated by two newlines.

## 2. Template Module

**Version:1.0.0**

### 2.1 Inputs

```
{
  "template_id": "string",   // Identifier for which document template to use
  "data": { … }              // Arbitrary key/value fields to populate the template
}
```

- template_id: e.g. "nda_v1", "contract_draft_01", etc.

- ata: a JSON object whose keys correspond to placeholder names inside the chosen template. Values can be strings, numbers, booleans, or nested objects, depending on the template’s design.

### 2.2 Outputs
```
{
  "generated_text": "string"  // Fully rendered document text, with placeholders filled
}
```
- generated_text: for example, a contract draft where every placeholder (e.g. {{party_name}}) has been replaced by data["party_name"], etc.

## 3. Logic/Compliance Module

**Version: 1.0.0**


### 3.1 Inputs
```
{
  "rules": [
    {
      "rule_id": "string",         // Unique identifier for this rule instance
      "type": "string",            // "clause_presence" | "numeric_range" | "regex_check"
      "parameters": {
        // if type == "clause_presence":
        //   { "keyword": "string" }
        //
        // if type == "numeric_range":
        //   { "field_name": "string", "min": number (optional), "max": number (optional) }
        //
        // if type == "regex_check":
        //   { "pattern": "string" }
      }
    }
    // (repeat for each rule)
  ],
  "draft_text": "string",          // Full document text to be evaluated. In practice, this is provided by template module
}
```
* Note: In the future, we plan to replace the structured “rules” array with a free-form AI query.

### 3.2 Outputs
```
{
  "results": [
    {
      "rule_id": "string",       // Echoed from input
      "type": "string",          // Echoed from input
      "passed": boolean,         // true if this rule passed
      "details": {
        // if type == "clause_presence":
        //   { "found": boolean, "locations": [ { "page": number, "offset": number } ] }
        //
        // if type == "numeric_range":
        //   { "field_name": "string", "extracted_value": number (if found), "within_range": boolean }
        //
        // if type == "regex_check":
        //   { "matches": [ { "match_text": "string", "start_offset": number } ] }
      },
      "message": "string"       // Human-readable summary, e.g. “Force Majeure found on page 3” or “Penalty 750000 exceeds max 500000”
    }
    // (repeat for each rule)
  ],
  "overall_passed": boolean      // true if all rules passed
}
```

# End of Module Contracts