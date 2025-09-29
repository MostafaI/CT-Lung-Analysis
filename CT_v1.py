from utils import run


while True:
    dicom_dir = input("Dicom Path (or type 'q' to quit): ").strip()
    if dicom_dir.lower() in ("q", "quit", "exit"):
        print("Exiting.")
        break
    if not dicom_dir:
        print("Empty path, try again.")
        continue
    run(dicom_dir)

