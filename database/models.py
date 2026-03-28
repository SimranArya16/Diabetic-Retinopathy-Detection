# database/models.py
from sqlalchemy import (Column, Integer, String, Float,
                        DateTime, Text, ForeignKey)
from sqlalchemy.orm import declarative_base, relationship
import datetime

Base = declarative_base()


class Patient(Base):
    __tablename__ = 'patients'

    id         = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(50), unique=True, nullable=False)
    name       = Column(String(100), default='Unknown')
    age        = Column(Integer, default=0)
    gender     = Column(String(10), default='Unknown')
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    predictions = relationship('Prediction', back_populates='patient')

    def __repr__(self):
        return f"<Patient {self.patient_id}>"


class Prediction(Base):
    __tablename__ = 'predictions'

    id              = Column(Integer, primary_key=True, autoincrement=True)
    patient_id      = Column(String(50), ForeignKey('patients.patient_id'))
    image_name      = Column(String(200))
    image_path      = Column(String(500))
    predicted_class = Column(Integer)
    predicted_label = Column(String(50))
    confidence      = Column(Float)
    risk_level      = Column(String(20))
    prob_no_dr      = Column(Float)
    prob_mild       = Column(Float)
    prob_moderate   = Column(Float)
    prob_severe     = Column(Float)
    prob_pdr        = Column(Float)
    gradcam_path    = Column(String(500))
    lime_path       = Column(String(500))
    active_regions  = Column(Text)
    created_at      = Column(DateTime, default=datetime.datetime.utcnow)

    patient = relationship('Patient', back_populates='predictions')
    report  = relationship('Report',  back_populates='prediction',
                           uselist=False)

    def __repr__(self):
        return f"<Prediction {self.predicted_label} ({self.confidence:.2f})>"


class Report(Base):
    __tablename__ = 'reports'

    id            = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey('predictions.id'))
    pdf_path      = Column(String(500))
    doctor_notes  = Column(Text, default='')
    created_at    = Column(DateTime, default=datetime.datetime.utcnow)

    prediction = relationship('Prediction', back_populates='report')

    def __repr__(self):
        return f"<Report {self.id}>"