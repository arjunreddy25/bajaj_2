# Bajaj Hackathon RAG API

A FastAPI-based Question Answering system for insurance policy documents using RAG (Retrieval-Augmented Generation) with Google Gemini and Pinecone.

## Features

- **FastAPI REST API** with proper authentication
- **RAG Pipeline** using Google Gemini 2.5 Flash and Pinecone vector store
- **Parallel Processing** for multiple questions
- **Document Processing** from PDF URLs
- **Optimized Performance** with caching and parallel execution
- **Production Ready** with error handling and health checks

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Fill in your API keys:

```env
GOOGLE_API_KEY=your_google_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
API_KEY=your-secret-api-key-here
```

### 3. Run the Application

```bash
python app.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Main Endpoint: POST `/hackrx/run`

**Authentication:** Bearer token required

**Request Format:**
```json
{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}
```

**Response Format:**
```json
{
    "answers": [
        "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
        "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
        "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
        "The policy has a specific waiting period of two (2) years for cataract surgery.",
        "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
        "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
        "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
        "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
        "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
        "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
    ]
}
```

### Additional Endpoints

- **GET `/health`** - Health check endpoint
- **GET `/`** - API information and available endpoints
- **GET `/docs`** - Interactive API documentation (Swagger UI)

## Usage Examples

### Using curl

```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-api-key-here" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
      "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
      "What is the waiting period for pre-existing diseases (PED) to be covered?"
    ]
  }'
```

### Using Python requests

```python
import requests
import json

url = "http://localhost:8000/hackrx/run"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-api-key-here"
}

data = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?"
    ]
}

response = requests.post(url, headers=headers, json=data)
print(json.dumps(response.json(), indent=2))
```

## Architecture

### RAG Pipeline

1. **Document Processing**: Downloads PDF from URL and chunks into semantic pieces
2. **Vector Embedding**: Uses Google Gemini embedding model to create vector representations
3. **Vector Storage**: Stores embeddings in Pinecone vector database
4. **Query Processing**: 
   - Embeds user questions
   - Retrieves relevant document chunks
   - Generates answers using Gemini 2.5 Flash
5. **Parallel Processing**: Handles multiple questions concurrently for optimal performance

### Performance Optimizations

- **Caching**: Document retrieval results are cached to avoid redundant API calls
- **Parallel Processing**: Multiple questions are processed simultaneously
- **Optimized Models**: Uses Gemini 2.5 Flash for faster inference
- **Efficient Chunking**: Smart document splitting for better retrieval

## Deployment

### Local Development

```bash
python app.py
```

### Production Deployment

For production deployment, consider using:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Invalid request format or document download failure
- **401 Unauthorized**: Invalid or missing API key
- **500 Internal Server Error**: Processing errors with detailed messages

## Monitoring

- **Health Check**: `/health` endpoint for monitoring
- **Logging**: Comprehensive logging for debugging
- **Performance Metrics**: Processing time tracking

## Security

- **API Key Authentication**: Bearer token required for all requests
- **Input Validation**: Pydantic models ensure data integrity
- **Error Sanitization**: Sensitive information is not exposed in error messages

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure all environment variables are set correctly
2. **Document Download Failures**: Check URL accessibility and network connectivity
3. **Pinecone Connection Issues**: Verify Pinecone API key and index configuration
4. **Memory Issues**: For large documents, consider increasing system memory

### Logs

Check the console output for detailed processing logs and error messages.

## License

This project is developed for the Bajaj Hackathon submission. 