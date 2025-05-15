import os
import cv2
import dlib
import numpy as np
import sqlite3
import datetime
import time
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import threading
import shutil
import pandas as pd
import csv
import queue

# Global variables
face_detector = dlib.get_frontal_face_detector()
shape_predictor = None
face_recognizer = None
DB_PATH = "attendance_system.db"
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL_PATH = "dlib_face_recognition_resnet_model_v1.dat"
FACE_DATA_PATH = "FaceData"
LEGACY_DATA_PATH = "TrainingImage"
LEGACY_DETAILS_PATH = "StudentDetails/StudentDetails.csv"
THRESHOLD = 0.5  # Face recognition threshold - lower means more strict


class Database:
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """Thread-safe singleton pattern to get database instance"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def __init__(self):
        """Initialize database and create tables if they don't exist"""
        self.create_tables()
    
    def get_connection(self):
        """Get a thread-local connection to the database"""
        conn = sqlite3.connect(DB_PATH)
        return conn
    
    def create_tables(self):
        """Create the required database tables if they don't exist"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Students table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            registration_date TEXT
        )
        ''')

        # Face encodings table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            encoding BLOB,
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
        ''')

        # Attendance table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            date TEXT,
            time TEXT,
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
        ''')
        conn.commit()
        conn.close()

    def add_student(self, student_id, name):
        """Add a new student to the database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        try:
            cursor.execute(
                "INSERT INTO students (id, name, registration_date) VALUES (?, ?, ?)",
                (student_id, name, current_date)
            )
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            # Student ID already exists
            conn.close()
            return False

    def add_face_encoding(self, student_id, encoding):
        """Save a face encoding for a student"""
        conn = self.get_connection()
        cursor = conn.cursor()
        encoding_bytes = encoding.tobytes()
        cursor.execute(
            "INSERT INTO face_encodings (student_id, encoding) VALUES (?, ?)",
            (student_id, encoding_bytes)
        )
        conn.commit()
        conn.close()

    def get_student_encodings(self, student_id=None):
        """Get all face encodings or for a specific student if ID is provided"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if student_id:
            cursor.execute(
                "SELECT student_id, encoding FROM face_encodings WHERE student_id=?",
                (student_id,)
            )
        else:
            cursor.execute("SELECT student_id, encoding FROM face_encodings")
            
        encodings = []
        for row in cursor.fetchall():
            student_id = row[0]
            encoding = np.frombuffer(row[1], dtype=np.float64)
            encodings.append((student_id, encoding))
        
        conn.close()
        return encodings

    def mark_attendance(self, student_id):
        """Mark attendance for a student"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        # Check if attendance already marked today
        cursor.execute(
            "SELECT id FROM attendance WHERE student_id=? AND date=?",
            (student_id, date_str)
        )
        
        result = cursor.fetchone() is None
        
        if result:
            cursor.execute(
                "INSERT INTO attendance (student_id, date, time) VALUES (?, ?, ?)",
                (student_id, date_str, time_str)
            )
            conn.commit()
        
        conn.close()
        return result

    def get_student_name(self, student_id):
        """Get student name from ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM students WHERE id=?", (student_id,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return result[0]
        return "Unknown"

    def get_today_attendance(self):
        """Get all attendance records for today"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        cursor.execute(
            """
            SELECT a.student_id, s.name, a.date, a.time
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE a.date = ?
            ORDER BY a.time DESC
            """,
            (today,)
        )
        
        results = cursor.fetchall()
        conn.close()
        return results

    def export_attendance(self, date, filepath):
        """Export attendance records for a specific date to CSV"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT a.student_id, s.name, a.date, a.time
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE a.date = ?
            ORDER BY a.time
            """,
            (date,)
        )
        records = cursor.fetchall()
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ID', 'Name', 'Date', 'Time'])
            writer.writerows(records)
        
        conn.close()
        return len(records)

    def get_all_students(self):
        """Get list of all registered students"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, name FROM students ORDER BY id")
        results = cursor.fetchall()
        
        conn.close()
        return results


class FaceProcessor:
    def __init__(self):
        """Initialize face recognition components"""
        self.db = Database.get_instance()
        self.detector = face_detector
        
        
        if os.path.exists(SHAPE_PREDICTOR_PATH):
            try:
                self.shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
                print("Shape predictor loaded successfully")
            except Exception as e:
                print(f"Error loading shape predictor: {e}")
                self.shape_predictor = None
        else:
            self.shape_predictor = None
            print(f"Warning: Shape predictor file not found at {SHAPE_PREDICTOR_PATH}")
    
    # Check if face recognition model exists and load it
        if os.path.exists(FACE_RECOGNITION_MODEL_PATH):
            try:
                self.face_recognizer = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)
                print("Face recognizer loaded successfully")
            except Exception as e:
                print(f"Error loading face recognizer: {e}")
                self.face_recognizer = None
        else:
            self.face_recognizer = None
            print(f"Warning: Face recognition model not found at {FACE_RECOGNITION_MODEL_PATH}")
    
    # Create directory for face data if not exists
        os.makedirs(FACE_DATA_PATH, exist_ok=True)
        
        # Create directory for student details if not exists
        os.makedirs(os.path.dirname(LEGACY_DETAILS_PATH), exist_ok=True)

    def _align_face(self, image, face):
        """Simplified version that just returns the face region without alignment"""
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        return image[y:y+h, x:x+w]

    def _preprocess_image(self, image):
        """Preprocess image for better recognition"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply histogram equalization for better contrast
        equalized = cv2.equalizeHist(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        
        return blurred

    def get_face_encoding(self, image, face):
        """Get face encoding from image"""
        if self.face_recognizer is None:
            print("Face recognizer not loaded")
            return None
        if self.shape_predictor is None:
            print("Shape predictor not loaded")
            return None
    
        try:
        # Get face region with bounds checking
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
        
        # Check if coordinates are valid
            if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
                print(f"Face coordinates out of bounds: x={x}, y={y}, w={w}, h={h}, image shape={image.shape}")
                return None
            
        # Check if image is not empty
            if image is None or image.size == 0:
                print("Input image is empty")
                return None
            
            face_img = image[y:y+h, x:x+w]
        
        # Verify face_img is not empty
            if face_img is None or face_img.size == 0:
                print("Extracted face region is empty")
                return None
        
        # Convert to dlib RGB format if needed
            if len(face_img.shape) == 2:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
            elif face_img.shape[2] == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
        # Create a dlib rectangle for the face region
            dlib_rect = dlib.rectangle(0, 0, face_img.shape[1], face_img.shape[0])
        
        # Get facial landmarks
            shape = self.shape_predictor(face_img, dlib_rect)
            
        # Get face encoding
            encoding = self.face_recognizer.compute_face_descriptor(face_img, shape)
        
            return np.array(encoding)
    
        except Exception as e:
            print(f"Error in get_face_encoding: {e}")
            return None

    def compare_faces(self, known_encoding, face_encoding):
        """Compare faces using Euclidean distance"""
        dist = np.linalg.norm(known_encoding - face_encoding)
        return dist, dist < THRESHOLD

    def capture_face_images(self, student_id, name, num_images=30):
        """Capture multiple face images for training"""
        cap = cv2.VideoCapture(0)
    
        # Check if camera opened successfully
        if not cap.isOpened():
            return False, "Could not open camera"
        
        # Create directory for student's face images
        student_dir = os.path.join(FACE_DATA_PATH, f"{student_id}_{name}")
        os.makedirs(student_dir, exist_ok=True)
    
        # Initialize variables
        image_count = 0
        start_time = time.time()
        processed_frames = 0
        skip_frames = 2  # Reduced from 5 to 2 to process more frames
    
        while image_count < num_images and time.time() - start_time < 60:  # Add timeout of 60 seconds
            ret, frame = cap.read()
            if not ret:
                continue
            
            processed_frames += 1
            # Display frame with instructions
            time_elapsed = time.time() - start_time
            cv2.putText(frame, f"Capturing: {image_count}/{num_images}", 
                  (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Move your face slightly for better training", 
                  (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
            # Process every other frame instead of every 5th frame
            if processed_frames % skip_frames == 0:
                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray, 1)  # Add 1 for upsampling to improve detection
            
                if len(faces) == 1:
                    # If exactly one face is detected
                    face = faces[0]
                
                    # Get face region with margin
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    margin = int(min(w, h) * 0.2)  # 20% margin
                    x = max(0, x - margin)
                    y = max(0, y - margin)
                    w = min(frame.shape[1] - x, w + 2*margin)
                    h = min(frame.shape[0] - y, h + 2*margin)
                
                    # Draw rectangle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                    # Save face image
                    face_img = gray[y:y+h, x:x+w]
                    timestamp = int(time.time() * 1000)
                    filename = os.path.join(student_dir, f"{timestamp}.jpg")
                    cv2.imwrite(filename, face_img)
                
                    # Get face encoding and save to database
                    try:
                        encoding = self.get_face_encoding(frame, face)
                        if encoding is not None:
                            self.db.add_face_encoding(student_id, encoding)
                            image_count += 1
                    except Exception as e:
                        print(f"Error processing face: {e}")
                        continue
        
            # Display frame with count
            cv2.putText(frame, f"Capturing: {image_count}/{num_images}", 
                  (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Move your face slightly for better training", 
                  (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('Capturing Faces', frame)
        
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
    
        if image_count == 0:
            return False, "Failed to capture any face images. Please try again with better lighting."
        elif image_count < num_images:
            return True, f"Captured {image_count} face images (less than requested {num_images}, but should be enough)"
        else:
            return True, f"Successfully captured {image_count} face images"

    def recognize_faces(self, callback):
        """Start face recognition process for attendance"""
        cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            return False, "Could not open camera"
        
        # Get all face encodings from database
        all_encodings = self.db.get_student_encodings()
        if not all_encodings:
            cap.release()
            return False, "No registered faces found in database"
        
        recognized_students = set()  # To avoid duplicate recognitions
        
        # Countdown before starting recognition
        countdown = 3
        while countdown > 0:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Display countdown
            cv2.putText(frame, f"Starting in {countdown}...", (50, 70), 
                     cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 2)
            cv2.imshow('Face Recognition', frame)
            
            # Wait for 1 second
            cv2.waitKey(1000)
            countdown -= 1
        
        start_time = time.time()
        
        # Recognition loop
        while time.time() - start_time < 30:  # Run for 30 seconds max
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detector(gray)
            
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                try:
                    # Get face encoding
                    face_encoding = self.get_face_encoding(frame, face)
                    if face_encoding is None:
                        continue
                        
                    # Find best match
                    best_match = None
                    min_distance = float('inf')
                    
                    for student_id, encoding in all_encodings:
                        distance, match = self.compare_faces(encoding, face_encoding)
                        if distance < min_distance:
                            min_distance = distance
                            if match:
                                best_match = student_id
                                
                    # If match found
                    if best_match and best_match not in recognized_students:
                        student_name = self.db.get_student_name(best_match)
                        
                        # Mark attendance
                        if self.db.mark_attendance(best_match):
                            recognized_students.add(best_match)
                            
                            # Update UI via callback
                            callback(best_match, student_name, min_distance)
                        
                        # Display name and confidence
                        confidence = max(0, min(100, int((1 - min_distance/0.6) * 100)))
                        label = f"{student_name} ({confidence}%)"
                        cv2.putText(frame, label, (x, y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        # Display "Unknown" or distance too high
                        cv2.putText(frame, "Unknown", (x, y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                except Exception as e:
                    print(f"Error in recognition: {e}")
                    continue
            
            # Show time remaining
            time_left = int(30 - (time.time() - start_time))
            cv2.putText(frame, f"Time left: {time_left}s", (20, 40), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to stop", (20, 80), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('Face Recognition', frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        return True, f"Recognition complete. {len(recognized_students)} students marked present."

    def import_legacy_data(self):
        """Import data from the legacy system (TrainingImage folder and CSV)"""
        if not os.path.exists(LEGACY_DETAILS_PATH):
            return False, "Legacy student details not found"
            
        try:
            # Read student details from CSV
            students_df = pd.read_csv(LEGACY_DETAILS_PATH)
            
            # Extract ID and Name columns
            student_data = []
            for _, row in students_df.iterrows():
                if 'SERIAL NO.' in row and 'ID' in row and 'NAME' in row:
                    student_id = row['ID']
                    name = row['NAME']
                    if pd.notna(student_id) and pd.notna(name):
                        student_data.append((int(student_id), name))
            
            # Add each student to database
            imported_count = 0
            for student_id, name in student_data:
                # Add student to database
                self.db.add_student(student_id, name)
                
                # Check for legacy images
                legacy_pattern = f"{name}.*{student_id}"
                legacy_images = []
                
                if os.path.exists(LEGACY_DATA_PATH):
                    for filename in os.listdir(LEGACY_DATA_PATH):
                        if str(student_id) in filename and (name.lower() in filename.lower() or 
                                                       name.replace(" ", ".").lower() in filename.lower()):
                            legacy_images.append(os.path.join(LEGACY_DATA_PATH, filename))
                
                # Process each legacy image
                for img_path in legacy_images:
                    try:
                        # Load image
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        
                        # Convert to grayscale
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        # Detect faces
                        faces = self.detector(gray)
                        if len(faces) == 1:
                            face = faces[0]
                            
                            # Get encoding and add to database
                            encoding = self.get_face_encoding(img, face)
                            if encoding is not None:
                                self.db.add_face_encoding(student_id, encoding)
                                imported_count += 1
                                
                                # Copy image to new faces directory
                                student_dir = os.path.join(FACE_DATA_PATH, f"{student_id}_{name}")
                                os.makedirs(student_dir, exist_ok=True)
                                new_img_path = os.path.join(student_dir, os.path.basename(img_path))
                                shutil.copy(img_path, new_img_path)
                    except Exception as e:
                        print(f"Error processing legacy image {img_path}: {e}")
                        continue
            
            return True, f"Imported {len(student_data)} students with {imported_count} face images"
            
        except Exception as e:
            return False, f"Error importing legacy data: {e}"
        
    def import_training_images(self):
        """Import training images from the TrainingImage folder"""
        if not os.path.exists(LEGACY_DATA_PATH):
            return False, "Training image folder not found"
            
        try:
            success_count = 0
            error_count = 0
            
            # Get all students from database
            all_students = self.db.get_all_students()
            student_dict = {str(student_id): name for student_id, name in all_students}
            
            # Print for debugging
            print(f"Found {len(student_dict)} students in database")
            print(f"Found {len(os.listdir(LEGACY_DATA_PATH))} files in training folder")
            
            # Process each image in the TrainingImage folder
            for filename in os.listdir(LEGACY_DATA_PATH):
                try:
                    print(f"Processing file: {filename}")
                
                    # Extract student ID from filename
                    student_id = None
                    # Try to find student ID in the filename
                    for id_str in student_dict.keys():
                        if id_str in filename:
                            student_id = id_str
                            print(f"Found student ID {student_id} in filename")
                            break
                
                    if student_id is None:
                        print(f"Could not extract student ID from {filename}")
                        error_count += 1
                        continue
                
                    # Check if student exists in database
                    if student_id not in student_dict:
                        print(f"Student ID {student_id} not found in database")
                        error_count += 1
                        continue
                
                    # Load and process image
                    img_path = os.path.join(LEGACY_DATA_PATH, filename)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Failed to load image: {img_path}")
                        error_count += 1
                        continue
                
                    # Convert to grayscale
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                    # Detect faces
                    faces = self.detector(gray)
                
                    if len(faces) == 0:
                        print(f"No faces detected in {filename}")
                        error_count += 1
                        continue
                
                    # Use first detected face
                    face = faces[0]
                
                    # Get encoding and add to database
                    try:
                        encoding = self.get_face_encoding(img, face)
                        if encoding is not None:
                            self.db.add_face_encoding(int(student_id), encoding)
                            success_count += 1
                        
                            # Copy image to new faces directory
                            student_name = student_dict[student_id]
                            student_dir = os.path.join(FACE_DATA_PATH, f"{student_id}_{student_name}")
                            os.makedirs(student_dir, exist_ok=True)
                            new_img_path = os.path.join(student_dir, os.path.basename(img_path))
                            shutil.copy(img_path, new_img_path)
                            print(f"Successfully processed {filename}")
                        else:
                            print(f"Failed to get encoding for {filename}")
                            error_count += 1
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
                        error_count += 1
            
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
                    error_count += 1
                    continue
        
            return True, f"Processed {success_count} face images successfully. {error_count} images failed."
        
        except Exception as e:
            print(f"Global exception: {e}")
            return False, f"Error importing training images: {e}"

    def update_csv_file(self, student_id, name):
        """Update or append to the CSV file with new student details"""
        os.makedirs(os.path.dirname(LEGACY_DETAILS_PATH), exist_ok=True)
        
        # Check if file exists
        file_exists = os.path.isfile(LEGACY_DETAILS_PATH)
        
        # Determine next serial number
        next_serial = 1
        if file_exists:
            try:
                df = pd.read_csv(LEGACY_DETAILS_PATH)
                if 'SERIAL NO.' in df.columns:
                    next_serial = df['SERIAL NO.'].max() + 1
            except Exception as e:
                print(f"Error reading CSV: {e}")
                
        # Write to CSV
        with open(LEGACY_DETAILS_PATH, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header if file is new
            if not file_exists:
                writer.writerow(['SERIAL NO.', 'ID', 'NAME'])
                
            # Write student info
            writer.writerow([next_serial, student_id, name])
            
        return True


class AttendanceSystemUI:
    def __init__(self, root):
        """Initialize the UI"""
        self.root = root
        self.root.title("Advanced Face Recognition Attendance System")
        self.root.geometry("1280x720")
        self.root.resizable(True, True)
        self.root.configure(background="#f0f0f0")
        
        # Initialize database
        self.db = Database.get_instance()
        
        # Initialize face processor
        self.face_processor = FaceProcessor()
        
        # Create UI elements
        self.create_ui()
        
        # Create a queue for thread-safe UI updates
        self.ui_queue = queue.Queue()
        self.process_ui_queue()
        
        # Check for legacy data
        if os.path.exists(LEGACY_DETAILS_PATH):
            self.ask_import_legacy_data()
            
    def get_date_time(self):
        """Get formatted date and time"""
        now = datetime.datetime.now()
        return now.strftime("%A, %d %B %Y, %H:%M:%S")
    
    def process_ui_queue(self):
        """Process queued UI updates"""
        try:
            while not self.ui_queue.empty():
                task = self.ui_queue.get(block=False)
                task()
        except queue.Empty:
            pass
        finally:
            # Schedule to run again
            self.root.after(100, self.process_ui_queue)
    
    def queue_ui_task(self, task):
        """Add a task to the UI queue"""
        self.ui_queue.put(task)

    def check_models(self):
        """Check if required model files exist, and if not, offer download instructions"""
        missing_files = []
        
        if not os.path.exists(SHAPE_PREDICTOR_PATH):
            missing_files.append(SHAPE_PREDICTOR_PATH)
            
        if not os.path.exists(FACE_RECOGNITION_MODEL_PATH):
            missing_files.append(FACE_RECOGNITION_MODEL_PATH)
            
        if missing_files:
            # Continues from the check_models method in AttendanceSystemUI class
            messagebox.showwarning(
                "Required Files Missing",
                f"The following files are missing:\n{', '.join(missing_files)}\n\n"
                "Please download these files from https://github.com/davisking/dlib-models:\n"
                "- shape_predictor_68_face_landmarks.dat\n"
                "- dlib_face_recognition_resnet_model_v1.dat\n\n"
                "Place them in the same directory as this program."
            )
    
    def create_ui(self):
        """Create the main UI elements"""
        # Create main frames
        self.title_frame = tk.Frame(self.root, bg="#2c3e50")
        self.title_frame.pack(fill=tk.X)
        
        self.main_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create title label
        self.title_label = tk.Label(
            self.title_frame, 
            text="Face Recognition Attendance System",
            font=("Arial", 20, "bold"),
            bg="#2c3e50",
            fg="white",
            pady=10
        )
        self.title_label.pack()
        
        # Create date-time label
        self.datetime_label = tk.Label(
            self.title_frame,
            text=self.get_date_time(),
            font=("Arial", 12),
            bg="#2c3e50",
            fg="white",
            pady=5
        )
        self.datetime_label.pack()
        
        # Update time every second
        self.update_time()
        
        # Create tabs
        self.tab_control = ttk.Notebook(self.main_frame)
        
        self.register_tab = tk.Frame(self.tab_control, bg="#f0f0f0")
        self.attendance_tab = tk.Frame(self.tab_control, bg="#f0f0f0")
        self.reports_tab = tk.Frame(self.tab_control, bg="#f0f0f0")
        self.settings_tab = tk.Frame(self.tab_control, bg="#f0f0f0")
        
        self.tab_control.add(self.register_tab, text="Register Students")
        self.tab_control.add(self.attendance_tab, text="Take Attendance")
        self.tab_control.add(self.reports_tab, text="Reports")
        self.tab_control.add(self.settings_tab, text="Settings")
        
        self.tab_control.pack(expand=1, fill="both")
        
        # Create registration form
        self.create_registration_tab()
        
        # Create attendance tab
        self.create_attendance_tab()
        
        # Create reports tab
        self.create_reports_tab()
        
        # Create settings tab
        self.create_settings_tab()
        
        # Check for required model files
        self.check_models()
    
    def update_time(self):
        """Update date and time label"""
        self.datetime_label.config(text=self.get_date_time())
        self.root.after(1000, self.update_time)
    
    def create_registration_tab(self):
        """Create the registration tab with form fields"""
        # Create form frame
        form_frame = tk.Frame(self.register_tab, bg="#f0f0f0", padx=20, pady=20)
        form_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create form title
        reg_title = tk.Label(
            form_frame,
            text="Register New Student",
            font=("Arial", 16, "bold"),
            bg="#f0f0f0",
            pady=10
        )
        reg_title.grid(row=0, column=0, columnspan=2, sticky="w")
        
        # Student ID
        tk.Label(
            form_frame,
            text="Student ID:",
            font=("Arial", 12),
            bg="#f0f0f0",
            pady=5
        ).grid(row=1, column=0, sticky="w")
        
        self.student_id_var = tk.StringVar()
        self.student_id_entry = tk.Entry(
            form_frame,
            textvariable=self.student_id_var,
            font=("Arial", 12),
            width=20
        )
        self.student_id_entry.grid(row=1, column=1, sticky="w", pady=5)
        
        # Student Name
        tk.Label(
            form_frame,
            text="Student Name:",
            font=("Arial", 12),
            bg="#f0f0f0",
            pady=5
        ).grid(row=2, column=0, sticky="w")
        
        self.student_name_var = tk.StringVar()
        self.student_name_entry = tk.Entry(
            form_frame,
            textvariable=self.student_name_var,
            font=("Arial", 12),
            width=20
        )
        self.student_name_entry.grid(row=2, column=1, sticky="w", pady=5)
        
        # Number of images to capture
        tk.Label(
            form_frame,
            text="Number of images:",
            font=("Arial", 12),
            bg="#f0f0f0",
            pady=5
        ).grid(row=3, column=0, sticky="w")
        
        self.num_images_var = tk.IntVar(value=30)
        self.num_images_entry = tk.Entry(
            form_frame,
            textvariable=self.num_images_var,
            font=("Arial", 12),
            width=10
        )
        self.num_images_entry.grid(row=3, column=1, sticky="w", pady=5)
        
        # Instructions
        instruction_text = (
            "Instructions:\n"
            "1. Enter the student's ID and name\n"
            "2. Click 'Capture Faces' to start camera\n"
            "3. The system will capture multiple face images\n"
            "4. Keep changing face angle slightly for better results\n"
            "5. Press 'q' to stop capturing early"
        )
        
        instructions = tk.Label(
            form_frame,
            text=instruction_text,
            font=("Arial", 10),
            bg="#f0f0f0",
            justify=tk.LEFT,
            pady=10
        )
        instructions.grid(row=4, column=0, columnspan=2, sticky="w")
        
        # Capture button
        self.register_button = tk.Button(
            form_frame,
            text="Capture Faces",
            font=("Arial", 12, "bold"),
            bg="#27ae60",
            fg="white",
            padx=10,
            pady=5,
            command=self.register_student
        )
        self.register_button.grid(row=5, column=0, columnspan=2, sticky="w", pady=10)
        
        # Status label
        self.register_status = tk.Label(
            form_frame,
            text="",
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="#e74c3c",
            wraplength=350
        )
        self.register_status.grid(row=6, column=0, columnspan=2, sticky="w")
        
        # Create a frame for the student list
        list_frame = tk.Frame(self.register_tab, bg="white", padx=20, pady=20)
        list_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create a title for the student list
        list_title = tk.Label(
            list_frame,
            text="Registered Students",
            font=("Arial", 16, "bold"),
            bg="white",
            pady=10
        )
        list_title.pack(anchor="w")
        
        # Create a treeview for the student list
        self.student_tree = ttk.Treeview(list_frame, columns=("ID", "Name"), show="headings")
        self.student_tree.heading("ID", text="Student ID")
        self.student_tree.heading("Name", text="Name")
        self.student_tree.column("ID", width=100)
        self.student_tree.column("Name", width=200)
        self.student_tree.pack(fill=tk.BOTH, expand=True)
        
        # Populate student list
        self.update_student_list()
    
    def create_attendance_tab(self):
        """Create the attendance tab"""
        # Create control frame
        control_frame = tk.Frame(self.attendance_tab, bg="#f0f0f0", padx=20, pady=20)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Create title
        att_title = tk.Label(
            control_frame,
            text="Take Attendance",
            font=("Arial", 16, "bold"),
            bg="#f0f0f0",
            pady=10
        )
        att_title.pack(anchor="w")
        
        # Instructions
        instruction_text = (
            "Instructions:\n"
            "1. Click 'Start Recognition' to open camera\n"
            "2. The system will automatically recognize faces\n"
            "3. Recognized students will be marked present\n"
            "4. Press 'q' to stop the camera at any time\n"
            "5. The process will automatically stop after 30 seconds"
        )
        
        instructions = tk.Label(
            control_frame,
            text=instruction_text,
            font=("Arial", 10),
            bg="#f0f0f0",
            justify=tk.LEFT,
            pady=10
        )
        instructions.pack(anchor="w")
        
        # Start button
        self.start_button = tk.Button(
            control_frame,
            text="Start Recognition",
            font=("Arial", 12, "bold"),
            bg="#3498db",
            fg="white",
            padx=10,
            pady=5,
            command=self.start_recognition
        )
        self.start_button.pack(anchor="w", pady=10)
        
        # Status label
        self.recognition_status = tk.Label(
            control_frame,
            text="",
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="#e74c3c",
            wraplength=350
        )
        self.recognition_status.pack(anchor="w")
        
        # Create a frame for the attendance list
        list_frame = tk.Frame(self.attendance_tab, bg="white", padx=20, pady=20)
        list_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create a title for the attendance list
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        list_title = tk.Label(
            list_frame,
            text=f"Today's Attendance ({today})",
            font=("Arial", 16, "bold"),
            bg="white",
            pady=10
        )
        list_title.pack(anchor="w")
        
        # Create a treeview for the attendance list
        self.attendance_tree = ttk.Treeview(
            list_frame, 
            columns=("ID", "Name", "Time", "Status"),
            show="headings"
        )
        self.attendance_tree.heading("ID", text="Student ID")
        self.attendance_tree.heading("Name", text="Name")
        self.attendance_tree.heading("Time", text="Time")
        self.attendance_tree.heading("Status", text="Status")
        self.attendance_tree.column("ID", width=80)
        self.attendance_tree.column("Name", width=200)
        self.attendance_tree.column("Time", width=100)
        self.attendance_tree.column("Status", width=100)
        self.attendance_tree.pack(fill=tk.BOTH, expand=True)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.attendance_tree.yview)
        self.attendance_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        
        # Refresh button
        refresh_button = tk.Button(
            list_frame,
            text="Refresh",
            font=("Arial", 10),
            bg="#7f8c8d",
            fg="white",
            padx=10,
            pady=5,
            command=self.update_attendance_list
        )
        refresh_button.pack(anchor="e", pady=10)
        
        # Populate attendance list
        self.update_attendance_list()
    
    def create_reports_tab(self):
        """Create the reports tab"""
        # Create the frame
        reports_frame = tk.Frame(self.reports_tab, bg="#f0f0f0", padx=20, pady=20)
        reports_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create title
        reports_title = tk.Label(
            reports_frame,
            text="Attendance Reports",
            font=("Arial", 16, "bold"),
            bg="#f0f0f0",
            pady=10
        )
        reports_title.grid(row=0, column=0, columnspan=3, sticky="w")
        
        # Date selection
        tk.Label(
            reports_frame,
            text="Select Date:",
            font=("Arial", 12),
            bg="#f0f0f0",
            pady=5
        ).grid(row=1, column=0, sticky="w")
        
        # Date entry components
        date_frame = tk.Frame(reports_frame, bg="#f0f0f0")
        date_frame.grid(row=1, column=1, sticky="w", pady=5)
        
        # Day
        self.day_var = tk.StringVar(value=datetime.datetime.now().strftime("%d"))
        self.day_combo = ttk.Combobox(
            date_frame,
            textvariable=self.day_var,
            width=3,
            values=[f"{i:02d}" for i in range(1, 32)]
        )
        self.day_combo.pack(side=tk.LEFT, padx=2)
        
        # Month
        self.month_var = tk.StringVar(value=datetime.datetime.now().strftime("%m"))
        self.month_combo = ttk.Combobox(
            date_frame,
            textvariable=self.month_var,
            width=3,
            values=[f"{i:02d}" for i in range(1, 13)]
        )
        self.month_combo.pack(side=tk.LEFT, padx=2)
        
        # Year
        self.year_var = tk.StringVar(value=datetime.datetime.now().strftime("%Y"))
        self.year_combo = ttk.Combobox(
            date_frame,
            textvariable=self.year_var,
            width=5,
            values=[str(year) for year in range(2020, 2030)]
        )
        self.year_combo.pack(side=tk.LEFT, padx=2)
        
        # Export button
        self.export_button = tk.Button(
            reports_frame,
            text="Export to CSV",
            font=("Arial", 12),
            bg="#9b59b6",
            fg="white",
            padx=10,
            pady=5,
            command=self.export_report
        )
        self.export_button.grid(row=1, column=2, sticky="w", padx=10)
        
        # Create a frame for the report list
        list_frame = tk.Frame(reports_frame, bg="white", pady=10)
        list_frame.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=10)
        reports_frame.grid_rowconfigure(2, weight=1)
        reports_frame.grid_columnconfigure(0, weight=1)
        reports_frame.grid_columnconfigure(1, weight=1)
        reports_frame.grid_columnconfigure(2, weight=1)
        
        # Create a treeview for the report list
        self.report_tree = ttk.Treeview(
            list_frame, 
            columns=("ID", "Name", "Date", "Time"),
            show="headings"
        )
        self.report_tree.heading("ID", text="Student ID")
        self.report_tree.heading("Name", text="Name")
        self.report_tree.heading("Date", text="Date")
        self.report_tree.heading("Time", text="Time")
        self.report_tree.column("ID", width=80)
        self.report_tree.column("Name", width=200)
        self.report_tree.column("Date", width=100)
        self.report_tree.column("Time", width=100)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.report_tree.yview)
        self.report_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.report_tree.pack(side="left", fill="both", expand=True)
        
        # Load today's report by default
        self.load_report()
    
    def create_settings_tab(self):
        """Create the settings tab"""
        # Create the frame
        settings_frame = tk.Frame(self.settings_tab, bg="#f0f0f0", padx=20, pady=20)
        settings_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create title
        settings_title = tk.Label(
            settings_frame,
            text="System Settings",
            font=("Arial", 16, "bold"),
            bg="#f0f0f0",
            pady=10
        )
        settings_title.grid(row=0, column=0, columnspan=2, sticky="w")
        
        # Recognition threshold
        tk.Label(
            settings_frame,
            text="Face Recognition Threshold:",
            font=("Arial", 12),
            bg="#f0f0f0",
            pady=5
        ).grid(row=1, column=0, sticky="w")
        
        self.threshold_var = tk.DoubleVar(value=THRESHOLD)
        threshold_scale = ttk.Scale(
            settings_frame,
            from_=0.3,
            to=0.7,
            orient="horizontal",
            variable=self.threshold_var,
            length=200
        )
        threshold_scale.grid(row=1, column=1, sticky="w", pady=5)
        
        threshold_label = tk.Label(
            settings_frame,
            textvariable=self.threshold_var,
            font=("Arial", 10),
            bg="#f0f0f0"
        )
        threshold_label.grid(row=1, column=2, sticky="w", padx=5)
        
        # Import TrainingImage button
        self.import_button = tk.Button(
            settings_frame,
            text="Import Training Images",
            font=("Arial", 12),
            bg="#e67e22",
            fg="white",
            padx=10,
            pady=5,
            command=self.import_training_images
        )
        self.import_button.grid(row=2, column=0, columnspan=2, sticky="w", pady=10)
        
        # About information
        about_frame = tk.LabelFrame(
            settings_frame,
            text="About",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
            padx=10,
            pady=10
        )
        about_frame.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=20)
        
        about_text = (
            "Advanced Face Recognition Attendance System\n"
            "Version 2.0\n\n"
            "This system uses Dlib's face recognition model for accurate face detection and recognition.\n"
            "The application automatically marks attendance when a student's face is recognized.\n\n"
            "Required Files:\n"
            "- shape_predictor_68_face_landmarks.dat\n"
            "- dlib_face_recognition_resnet_model_v1.dat"
        )
        
        about_label = tk.Label(
            about_frame,
            text=about_text,
            font=("Arial", 10),
            bg="#f0f0f0",
            justify=tk.LEFT,
            wraplength=600
        )
        about_label.pack(anchor="w")
        
        # Save settings button
        self.save_settings_button = tk.Button(
            settings_frame,
            text="Save Settings",
            font=("Arial", 12),
            bg="#2980b9",
            fg="white",
            padx=10,
            pady=5,
            command=self.save_settings
        )
        self.save_settings_button.grid(row=4, column=0, columnspan=2, sticky="w", pady=10)
        
        # Status label
        self.settings_status = tk.Label(
            settings_frame,
            text="",
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="#27ae60",
            wraplength=400
        )
        self.settings_status.grid(row=5, column=0, columnspan=3, sticky="w")
    
    def ask_import_legacy_data(self):
        """Ask user if they want to import legacy data"""
        result = messagebox.askyesno(
            "Legacy Data Found",
            "Student details found from previous version. Would you like to import this data?"
        )
        
        if result:
            threading.Thread(target=self.import_legacy_data).start()
    
    def import_legacy_data(self):
        """Import data from legacy format"""
        # Update UI to show we're importing
        self.queue_ui_task(lambda: self.settings_status.config(
            text="Importing legacy data... Please wait.",
            fg="#e67e22"
        ))
        
        # Import the data
        success, message = self.face_processor.import_legacy_data()
        
        # Update UI with result
        color = "#27ae60" if success else "#e74c3c"
        self.queue_ui_task(lambda: self.settings_status.config(text=message, fg=color))
        
        # Refresh student list
        if success:
            self.queue_ui_task(self.update_student_list)
    
    def update_student_list(self):
        """Update the list of registered students"""
        # Clear the current list
        for item in self.student_tree.get_children():
            self.student_tree.delete(item)
        
        # Get all students from database
        students = self.db.get_all_students()
        
        # Add students to the list
        for student_id, name in students:
            self.student_tree.insert("", "end", values=(student_id, name))
    
    def update_attendance_list(self):
        """Update the list of today's attendance"""
        # Clear the current list
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)
        
        # Get today's attendance from database
        attendance = self.db.get_today_attendance()
        
        # Add attendance records to the list
        for student_id, name, date, time in attendance:
            self.attendance_tree.insert("", 0, values=(student_id, name, time, "Present"))
    
    def load_report(self):
        """Load attendance report for selected date"""
        # Clear the current list
        for item in self.report_tree.get_children():
            self.report_tree.delete(item)
        
        # Get selected date
        try:
            day = self.day_var.get()
            month = self.month_var.get()
            year = self.year_var.get()
            date_str = f"{year}-{month}-{day}"
            date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Invalid Date", "Please enter a valid date")
            return
        
        # Connect to database
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        # Get attendance records for the selected date
        cursor.execute(
            """
            SELECT a.student_id, s.name, a.date, a.time
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE a.date = ?
            ORDER BY a.time
            """,
            (date_str,)
        )
        
        records = cursor.fetchall()
        conn.close()
        
        # Add records to the tree
        for student_id, name, date, time in records:
            self.report_tree.insert("", "end", values=(student_id, name, date, time))
    
    def register_student(self):
        """Register a new student with face images"""
        # Get the student details
        student_id = self.student_id_var.get().strip()
        name = self.student_name_var.get().strip()
        num_images = self.num_images_var.get()
        
        # Validate the inputs
        if not student_id or not name:
            self.register_status.config(
                text="Please enter both Student ID and Name",
                fg="#e74c3c"
            )
            return
        
        try:
            student_id = int(student_id)
        except ValueError:
            self.register_status.config(
                text="Student ID must be a number",
                fg="#e74c3c"
            )
            return
        
        # Update status
        self.register_status.config(
            text="Registering student... Please wait.",
            fg="#e67e22"
        )
        self.register_button.config(state=tk.DISABLED)
        
        # Run face capture in a separate thread
        threading.Thread(target=self.capture_student_faces, 
                        args=(student_id, name, num_images)).start()
    
    def capture_student_faces(self, student_id, name, num_images):
        """Capture student faces in a separate thread"""
        # Add student to database
        if not self.db.add_student(student_id, name):
            self.queue_ui_task(lambda: self.register_status.config(
                text=f"Student ID {student_id} already exists",
                fg="#e74c3c"
            ))
            self.queue_ui_task(lambda: self.register_button.config(state=tk.NORMAL))
            return
        
        # Update CSV file for legacy compatibility
        self.face_processor.update_csv_file(student_id, name)
        
        # Capture faces
        success, message = self.face_processor.capture_face_images(student_id, name, num_images)
        
        # Update UI with result
        color = "#27ae60" if success else "#e74c3c"
        self.queue_ui_task(lambda: self.register_status.config(text=message, fg=color))
        self.queue_ui_task(lambda: self.register_button.config(state=tk.NORMAL))
        
        # Update student list
        if success:
            self.queue_ui_task(self.update_student_list)
            
            # Clear form fields
            self.queue_ui_task(lambda: self.student_id_var.set(""))
            self.queue_ui_task(lambda: self.student_name_var.set(""))
    
    def start_recognition(self):
        """Start face recognition for attendance"""
        # Disable the button to prevent multiple clicks
        self.start_button.config(state=tk.DISABLED)
        self.recognition_status.config(
            text="Starting face recognition... Please wait.",
            fg="#e67e22"
        )
        
        # Run recognition in a separate thread
        threading.Thread(target=self.recognize_faces).start()
    
    def recognize_faces(self):
        """Run face recognition in a separate thread"""
        # Callback function to update UI from the recognition thread
        def recognition_callback(student_id, name, confidence):
            self.queue_ui_task(lambda: self.attendance_tree.insert(
                "", 0, values=(student_id, name, datetime.datetime.now().strftime("%H:%M:%S"), "Present")
            ))
        
        # Start recognition
        success, message = self.face_processor.recognize_faces(recognition_callback)
        
        # Update UI with result
        color = "#27ae60" if success else "#e74c3c"
        self.queue_ui_task(lambda: self.recognition_status.config(text=message, fg=color))
        self.queue_ui_task(lambda: self.start_button.config(state=tk.NORMAL))
    
    def export_report(self):
        """Export attendance report to CSV"""
        # Get selected date
        try:
            day = self.day_var.get()
            month = self.month_var.get()
            year = self.year_var.get()
            date_str = f"{year}-{month}-{day}"
            date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            formatted_date = date_obj.strftime("%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Invalid Date", "Please enter a valid date")
            return
        
        # Ask for save location
        from tkinter import filedialog
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"Attendance_{formatted_date}.csv"
        )
        
        if not filepath:
            return
        
        # Export the report
        try:
            count = self.db.export_attendance(date_str, filepath)
            messagebox.showinfo(
                "Export Successful",
                f"Successfully exported {count} attendance records to {filepath}"
            )
        except Exception as e:
            messagebox.showerror("Export Failed", f"Error exporting report: {e}")
    
    def import_training_images(self):
        """Import training images from the TrainingImage folder"""
        # Update UI to show we're importing
        self.settings_status.config(
            text="Importing training images... Please wait.",
            fg="#e67e22"
        )
        self.import_button.config(state=tk.DISABLED)
        
        # Run import in a separate thread
        threading.Thread(target=self.run_import_training_images).start()
    
    def run_import_training_images(self):
        """Run the import in a separate thread"""
        # Import the data
        success, message = self.face_processor.import_training_images()
        
        # Update UI with result
        color = "#27ae60" if success else "#e74c3c"
        self.queue_ui_task(lambda: self.settings_status.config(text=message, fg=color))
        self.queue_ui_task(lambda: self.import_button.config(state=tk.NORMAL))
    
    def save_settings(self):
        """Save the application settings"""
        global THRESHOLD
        
        # Update threshold
        THRESHOLD = self.threshold_var.get()
        
        # Show success message
        self.settings_status.config(
        text="Settings saved successfully",
        fg="#27ae60"
    )
    
    # You might want to save to a config file for persistence
    # This is a simple implementation

# Main execution
if __name__ == "__main__":
    # Create root window
    root = tk.Tk()
    
    # Create app
    app = AttendanceSystemUI(root)
    
    # Start main loop
    root.mainloop()