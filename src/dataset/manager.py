import os
import pandas as pd
import PyPDF2
from PIL import Image
from pathlib import Path
from collections import defaultdict
import logging
import json
from datetime import datetime
import xml.etree.ElementTree as ET
import mimetypes

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Direktori dataset
DATASET_DIR = Path("D:/athala-adjutor/data/raw/new_dataset")
OUTPUT_DIR = Path("D:/athala-adjutor/data/processed")

class DatasetManager:
    def __init__(self, dataset_dir=DATASET_DIR, output_dir=OUTPUT_DIR):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_types = defaultdict(list)
        self.total_size = 0
        self.file_count = 0

    def scan_dataset(self):
        """Scan dataset dan kelompokkan file berdasarkan ekstensi."""
        logger.info(f"Memindai direktori: {self.dataset_dir}")
        for root, dirs, files in os.walk(self.dataset_dir):
            for file in files:
                file_path = Path(root) / file
                ekstensi = file_path.suffix.lower()
                self.file_types[ekstensi].append(str(file_path))
                self.total_size += file_path.stat().st_size
                self.file_count += 1
        logger.info(f"Selesai memindai. Total file: {self.file_count}, Ukuran: {self.total_size / (1024**3):.2f} GB")

    def process_csv(self, file_path):
        """Proses file CSV dan simpan ringkasan."""
        try:
            df = pd.read_csv(file_path, nrows=1000)  # Baca 1000 baris dulu
            summary = {
                "file": str(file_path),
                "rows": len(df),
                "columns": list(df.columns),
                "sample_data": df.head(5).to_dict(orient="records")
            }
            output_file = self.output_dir / f"summary_{Path(file_path).name}.json"
            with open(output_file, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Ringkasan CSV disimpan: {output_file}")
            return summary
        except Exception as e:
            logger.error(f"Gagal proses {file_path}: {e}")
            return {"file": str(file_path), "status": "failed", "error": str(e)}

    def process_pdf(self, file_path):
        """Ekstrak teks dari PDF (maksimum 500 halaman) dan simpan."""
        try:
            with open(file_path, "rb") as f:
                pdf = PyPDF2.PdfReader(f)
                total_pages = len(pdf.pages)
                max_pages = min(total_pages, 500)  # Baca maksimum 500 halaman
                text = ""
                for page in pdf.pages[:max_pages]:
                    extracted_text = page.extract_text() or ""
                    text += extracted_text
                output_file = self.output_dir / f"extracted_{Path(file_path).name}.txt"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(text)  # Simpan seluruh teks yang diekstrak
                logger.info(f"Teks PDF disimpan: {output_file}, Halaman diproses: {max_pages}/{total_pages}")
                return {
                    "file": str(file_path),
                    "extracted_length": len(text),
                    "total_pages": total_pages,
                    "processed_pages": max_pages
                }
        except Exception as e:
            logger.error(f"Gagal proses {file_path}: {e}")
            return {"file": str(file_path), "status": "failed", "error": str(e)}

    def process_image(self, file_path):
        """Ambil metadata gambar dan simpan."""
        try:
            with Image.open(file_path) as img:
                metadata = {
                    "file": str(file_path),
                    "format": img.format,
                    "size": img.size,
                    "mode": img.mode
                }
            output_file = self.output_dir / f"metadata_{Path(file_path).name}.json"
            with open(output_file, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata gambar disimpan: {output_file}")
            return metadata
        except Exception as e:
            logger.error(f"Gagal proses {file_path}: {e}")
            return {"file": str(file_path), "status": "failed", "error": str(e)}

    def process_jsonl(self, file_path):
        """Proses file JSONL dan simpan ringkasan."""
        try:
            with open(file_path, "r") as f:
                lines = [json.loads(line.strip()) for line in f.readlines()[:1000]]  # Baca 1000 baris
            summary = {
                "file": str(file_path),
                "rows": len(lines),
                "sample_data": lines[:5]
            }
            output_file = self.output_dir / f"summary_{Path(file_path).name}.json"
            with open(output_file, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Ringkasan JSONL disimpan: {output_file}")
            return summary
        except Exception as e:
            logger.error(f"Gagal proses {file_path}: {e}")
            return {"file": str(file_path), "status": "failed", "error": str(e)}

    def process_xml(self, file_path):
        """Proses file XML dan simpan ringkasan."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            summary = {
                "file": str(file_path),
                "root_tag": root.tag,
                "child_tags": [child.tag for child in root][:5]
            }
            output_file = self.output_dir / f"summary_{Path(file_path).name}.json"
            with open(output_file, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Ringkasan XML disimpan: {output_file}")
            return summary
        except Exception as e:
            logger.error(f"Gagal proses {file_path}: {e}")
            return {"file": str(file_path), "status": "failed", "error": str(e)}

    def process_text(self, file_path):
        """Proses file teks (txt, md, conf, yml) dan simpan ringkasan."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(1000)  # Baca 1000 karakter pertama
            summary = {
                "file": str(file_path),
                "char_count": len(content),
                "sample_content": content[:200]
            }
            output_file = self.output_dir / f"summary_{Path(file_path).name}.json"
            with open(output_file, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Ringkasan teks disimpan: {output_file}")
            return summary
        except Exception as e:
            logger.error(f"Gagal proses {file_path}: {e}")
            return {"file": str(file_path), "status": "failed", "error": str(e)}

    def process_all(self, limit_per_type=None):
        """Proses semua file berdasarkan tipe."""
        self.scan_dataset()
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_files": self.file_count,
            "total_size_gb": self.total_size / (1024**3),
            "file_types": {ext: len(files) for ext, files in self.file_types.items()},
            "processed_files": []
        }

        for ext, files in self.file_types.items():
            # Terapkan batasan jumlah file per tipe kalau ada
            files_to_process = files if limit_per_type is None else files[:limit_per_type]
            for file in files_to_process:
                if ext == ".csv":
                    result = self.process_csv(file)
                elif ext == ".pdf":
                    result = self.process_pdf(file)
                elif ext in (".jpg", ".png"):
                    result = self.process_image(file)
                elif ext == ".jsonl":
                    result = self.process_jsonl(file)
                elif ext == ".xml":
                    result = self.process_xml(file)
                elif ext in (".txt", ".md", ".conf", ".yml", ".yar", ".ps1", ".html"):
                    result = self.process_text(file)
                else:
                    # Coba deteksi tipe file pake mimetypes
                    mime_type, _ = mimetypes.guess_type(file)
                    result = {
                        "file": file,
                        "status": "skipped",
                        "mime_type": mime_type or "unknown"
                    }
                report["processed_files"].append(result)

        # Simpan laporan
        report_file = self.output_dir / "dataset_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Laporan disimpan: {report_file}")
        return report

    def get_summary(self):
        """Tampilkan ringkasan dataset."""
        self.scan_dataset()
        summary = f"""
Direktori Dataset: {self.dataset_dir}
Total File: {self.file_count}
Total Ukuran: {self.total_size / (1024**3):.2f} GB
Tipe File:
"""
        for ext, files in self.file_types.items():
            summary += f"{ext or 'Tanpa Ekstensi'}: {len(files)} file\n"
        return summary

def main(component="summary", limit=None):
    """Jalankan pengelolaan dataset berdasarkan komponen."""
    os.chdir("D:/athala-adjutor")  # Set root proyek biar impor aman
    manager = DatasetManager()
    
    if component == "all":
        report = manager.process_all(limit_per_type=limit)
        print(json.dumps(report, indent=2))
    else:
        print(manager.get_summary())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kelola dataset AthalaSIEM")
    parser.add_argument("--component", default="summary", choices=["summary", "all"],
                        help="Komponen untuk dijalankan: summary atau all")
    parser.add_argument("--limit", type=int, default=None,
                        help="Batas jumlah file per tipe yang diproses (kosongkan untuk semua)")
    args = parser.parse_args()
    main(args.component, args.limit)