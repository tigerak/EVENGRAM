import uvicorn
from app import main_app

    
if __name__ == '__main__':
    uvicorn.run(
        "app:main_app",
        host="0.0.0.0",
        port=80,
        reload=True,
        workers=1,
    )