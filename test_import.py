from app import create_app
from app.models import db
from app.models.dataset import Dataset
from app.services.dataset_service import import_from_file

app = create_app()
with app.app_context():
    print("Menghapus data lama...")
    Dataset.query.delete()
    db.session.commit()
    
    print("Mengimport dataset baru...")
    class MockFile:
        def __init__(self, path):
            self.filename = "dataset.xlsx"
            self.path = path
        def read(self):
            with open(self.path, 'rb') as f:
                return f.read()

    file = MockFile(r"c:\Users\fauza\Documents\Kuliah\semester-7\PPAS\app_skripsi\diagnosis_kerusakan_dinamo\dataset\dataset.xlsx")
    result = import_from_file(file)
    print("Result:", result)
    print("Total data di database sekarang:", Dataset.query.count())
