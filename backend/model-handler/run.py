import uvicorn
from api import app  # Changed this line

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)