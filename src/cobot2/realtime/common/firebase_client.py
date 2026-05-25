import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

from common import settings


def initialize_firebase():
    settings.validate_firebase_settings()

    if firebase_admin._apps:
        return firebase_admin.get_app()

    cred = credentials.Certificate(settings.FIREBASE_SERVICE_ACCOUNT)

    return firebase_admin.initialize_app(
        cred,
        {
            "databaseURL": settings.FIREBASE_DATABASE_URL,
            "storageBucket": settings.FIREBASE_STORAGE_BUCKET,
        },
    )


def get_db_reference(path: str = "/"):
    initialize_firebase()
    return db.reference(path)


def get_storage_bucket():
    initialize_firebase()
    return storage.bucket()
