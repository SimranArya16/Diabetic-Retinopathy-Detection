# database/manage_db.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import SessionLocal, init_db
from database.models import Patient, Prediction, Report


init_db()
db = SessionLocal()


# ════════════════════════════════════════
#  ADD
# ════════════════════════════════════════

def add_patient(patient_id, name, age, gender):
    existing = db.query(Patient).filter(
        Patient.patient_id == patient_id).first()
    if existing:
        print(f"Patient {patient_id} already exists.")
        return existing
    p = Patient(patient_id=patient_id, name=name,
                age=age, gender=gender)
    db.add(p)
    db.commit()
    db.refresh(p)
    print(f"Added patient: {patient_id} — {name}")
    return p


# ════════════════════════════════════════
#  UPDATE
# ════════════════════════════════════════

def update_patient(patient_id, name=None, age=None, gender=None):
    p = db.query(Patient).filter(
        Patient.patient_id == patient_id).first()
    if not p:
        print(f"Patient {patient_id} not found.")
        return
    if name:   p.name   = name
    if age:    p.age    = age
    if gender: p.gender = gender
    db.commit()
    print(f"Updated patient: {patient_id}")


def update_doctor_notes(prediction_id, notes):
    r = db.query(Report).filter(
        Report.prediction_id == prediction_id).first()
    if not r:
        # Create report record if not exists
        r = Report(prediction_id=prediction_id, doctor_notes=notes)
        db.add(r)
    else:
        r.doctor_notes = notes
    db.commit()
    print(f"Updated notes for prediction ID: {prediction_id}")


# ════════════════════════════════════════
#  DELETE
# ════════════════════════════════════════

def delete_patient(patient_id):
    p = db.query(Patient).filter(
        Patient.patient_id == patient_id).first()
    if not p:
        print(f"Patient {patient_id} not found.")
        return
    # Delete related predictions first
    db.query(Prediction).filter(
        Prediction.patient_id == patient_id).delete()
    db.delete(p)
    db.commit()
    print(f"Deleted patient: {patient_id} and all their predictions.")


def delete_prediction(prediction_id):
    pred = db.query(Prediction).filter(
        Prediction.id == prediction_id).first()
    if not pred:
        print(f"Prediction ID {prediction_id} not found.")
        return
    # Delete related report first
    db.query(Report).filter(
        Report.prediction_id == prediction_id).delete()
    db.delete(pred)
    db.commit()
    print(f"Deleted prediction ID: {prediction_id}")


def delete_all_predictions():
    db.query(Report).delete()
    db.query(Prediction).delete()
    db.commit()
    print("Deleted all predictions and reports.")


def delete_all():
    db.query(Report).delete()
    db.query(Prediction).delete()
    db.query(Patient).delete()
    db.commit()
    print("Deleted everything from database.")


# ════════════════════════════════════════
#  VIEW
# ════════════════════════════════════════

def view_all_patients():
    patients = db.query(Patient).all()
    print("\n=== All Patients ===")
    if not patients:
        print("No patients found.")
        return
    for p in patients:
        print(f"  ID: {p.id} | Patient ID: {p.patient_id} | "
              f"Name: {p.name} | Age: {p.age} | Gender: {p.gender}")


def view_all_predictions():
    preds = db.query(Prediction).all()
    print("\n=== All Predictions ===")
    if not preds:
        print("No predictions found.")
        return
    for p in preds:
        print(f"  ID: {p.id} | Patient: {p.patient_id} | "
              f"Label: {p.predicted_label} | "
              f"Confidence: {p.confidence*100:.1f}% | "
              f"Date: {p.created_at}")


def view_patient(patient_id):
    p = db.query(Patient).filter(
        Patient.patient_id == patient_id).first()
    if not p:
        print(f"Patient {patient_id} not found.")
        return
    print(f"\n=== Patient: {patient_id} ===")
    print(f"  Name   : {p.name}")
    print(f"  Age    : {p.age}")
    print(f"  Gender : {p.gender}")
    preds = db.query(Prediction).filter(
        Prediction.patient_id == patient_id).all()
    print(f"  Total predictions: {len(preds)}")
    for pred in preds:
        print(f"    - {pred.predicted_label} "
              f"({pred.confidence*100:.1f}%) on {pred.created_at}")


# ════════════════════════════════════════
#  MENU
# ════════════════════════════════════════

def menu():
    while True:
        print("\n=============================")
        print("  DR Database Manager")
        print("=============================")
        print("1. View all patients")
        print("2. View all predictions")
        print("3. View specific patient")
        print("4. Add patient")
        print("5. Update patient")
        print("6. Update doctor notes")
        print("7. Delete patient")
        print("8. Delete prediction")
        print("9. Delete all predictions")
        print("10. Delete everything")
        print("0. Exit")
        print("=============================")

        choice = input("Enter choice: ").strip()

        if choice == '1':
            view_all_patients()

        elif choice == '2':
            view_all_predictions()

        elif choice == '3':
            pid = input("Enter Patient ID: ").strip()
            view_patient(pid)

        elif choice == '4':
            pid    = input("Patient ID : ").strip()
            name   = input("Name       : ").strip()
            age    = int(input("Age        : ").strip())
            gender = input("Gender (Male/Female/Unknown): ").strip()
            add_patient(pid, name, age, gender)

        elif choice == '5':
            pid    = input("Patient ID to update: ").strip()
            name   = input("New name (Enter to skip): ").strip() or None
            age    = input("New age  (Enter to skip): ").strip()
            age    = int(age) if age else None
            gender = input("New gender (Enter to skip): ").strip() or None
            update_patient(pid, name, age, gender)

        elif choice == '6':
            pid   = int(input("Prediction ID: ").strip())
            notes = input("Doctor notes : ").strip()
            update_doctor_notes(pid, notes)

        elif choice == '7':
            pid = input("Patient ID to delete: ").strip()
            confirm = input(f"Delete {pid}? (yes/no): ").strip()
            if confirm == 'yes':
                delete_patient(pid)

        elif choice == '8':
            pid = int(input("Prediction ID to delete: ").strip())
            confirm = input(f"Delete prediction {pid}? (yes/no): ").strip()
            if confirm == 'yes':
                delete_prediction(pid)

        elif choice == '9':
            confirm = input("Delete ALL predictions? (yes/no): ").strip()
            if confirm == 'yes':
                delete_all_predictions()

        elif choice == '10':
            confirm = input("Delete EVERYTHING? (yes/no): ").strip()
            if confirm == 'yes':
                delete_all()

        elif choice == '0':
            db.close()
            print("Bye!")
            break

        else:
            print("Invalid choice. Try again.")


if __name__ == '__main__':
    menu()