# ai-chatbot-pbe

## Admin document ingestion API

This project supports uploading internal business documents for later retrieval and RAG usage.

### Endpoint

POST /api/admin/knowledge/documents

### Request

- Content-Type: multipart/form-data
- Field name: file
- Supported file types: PDF, DOCX, TXT, XLSX

### Example

```bash
curl -X POST "http://localhost:8000/api/admin/knowledge/documents" \
  -F "file=@/path/to/your/document.pdf"
```

### Example success response

```json
{
  "document_id": "4f2a1ef9-1d0d-4a18-9d84-92bd5d3ef6ad",
  "filename": "document.pdf",
  "chunks_created": 12,
  "status": "completed"
}
```

### Notes

- The endpoint is intended for internal/admin usage only.
- The uploaded file is validated, cleaned, chunked, embedded, and stored in MongoDB for vector retrieval.
- Excel support is for .xlsx files and converts workbook rows into searchable text for the same chunking pipeline.
- This is a prototype RAG ingestion flow and is separate from the normal chat API.
