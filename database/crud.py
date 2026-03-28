# database/crud.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
from sqlalchemy.orm import Session
from database.models import Patient, Prediction, Report


# ── Patient operations ────────────────────────────────────────────────────────

def create_patient(db: Session, patient_id: str, name: str = 'Unknown',
                   age: int = 0, gender: str = 'Unknown'):
    existing = db.query(Patient).filter(
        Patient.patient_id == patient_id).first()
    if existing:
        return existing
    patient = Patient(patient_id=patient_id, name=name,
                      age=age, gender=gender)
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


def get_patient(db: Session, patient_id: str):
    return db.query(Patient).filter(
        Patient.patient_id == patient_id).first()


def get_all_patients(db: Session):
    return db.query(Patient).order_by(Patient.created_at.desc()).all()


# ── Prediction operations ─────────────────────────────────────────────────────

def save_prediction(db: Session, patient_id: str, image_name: str,
                    image_path: str, predicted_class: int,
                    predicted_label: str, confidence: float,
                    risk_level: str, probs: list,
                    gradcam_path: str = '', lime_path: str = '',
                    active_regions: list = None):
    pred = Prediction(
        patient_id      = patient_id,
        image_name      = image_name,
        image_path      = image_path,
        predicted_class = predicted_class,
        predicted_label = predicted_label,
        confidence      = round(float(confidence), 4),
        risk_level      = risk_level,
        prob_no_dr      = round(float(probs[0]), 4),
        prob_mild       = round(float(probs[1]), 4),
        prob_moderate   = round(float(probs[2]), 4),
        prob_severe     = round(float(probs[3]), 4),
        prob_pdr        = round(float(probs[4]), 4),
        gradcam_path    = gradcam_path,
        lime_path       = lime_path,
        active_regions  = ', '.join(active_regions) if active_regions else '',
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred


def get_predictions_by_patient(db: Session, patient_id: str):
    return db.query(Prediction).filter(
        Prediction.patient_id == patient_id
    ).order_by(Prediction.created_at.desc()).all()


def get_all_predictions(db: Session, limit: int = 100):
    return db.query(Prediction).order_by(
        Prediction.created_at.desc()).limit(limit).all()


def get_prediction_by_id(db: Session, pred_id: int):
    return db.query(Prediction).filter(Prediction.id == pred_id).first()


# ── Report operations ─────────────────────────────────────────────────────────

def save_report(db: Session, prediction_id: int,
                pdf_path: str, doctor_notes: str = ''):
    report = Report(prediction_id=prediction_id,
                    pdf_path=pdf_path,
                    doctor_notes=doctor_notes)
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def get_report_by_prediction(db: Session, prediction_id: int):
    return db.query(Report).filter(
        Report.prediction_id == prediction_id).first()


# ── Statistics ────────────────────────────────────────────────────────────────

def get_statistics(db: Session):
    from sqlalchemy import func
    total       = db.query(Prediction).count()
    by_class    = db.query(
        Prediction.predicted_label,
        func.count(Prediction.id)
    ).group_by(Prediction.predicted_label).all()
    avg_conf    = db.query(func.avg(Prediction.confidence)).scalar()
    total_pats  = db.query(Patient).count()

    return {
        'total_predictions': total,
        'total_patients':    total_pats,
        'avg_confidence':    round(float(avg_conf or 0), 4),
        'by_class':          {label: count for label, count in by_class},
    }