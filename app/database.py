from sqlalchemy import create_engine, Column, String, Integer, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String, unique=True, index=True)
    title = Column(String)
    channel = Column(String)
    published_at = Column(String)
    description = Column(String)
    comments = relationship("Comment", back_populates="video")

class Comment(Base):
    __tablename__ = "comments"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String, ForeignKey("videos.video_id"), index=True)
    author = Column(String)
    text = Column(Text)
    likes = Column(Integer, default=0)
    published_at = Column(String)
    sentiment = Column(String, nullable=True)  # positive / negative / neutral
    video = relationship("Video", back_populates="comments")

def init_db():
    Base.metadata.create_all(bind=engine)