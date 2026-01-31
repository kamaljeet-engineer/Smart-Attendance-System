                Smart Attendance System Using Face Recognition                               
                
📖 Description

The Smart Attendance System Using Face Recognition is an automated attendance management solution that uses computer vision and face recognition techniques to mark attendance in real time.
Instead of manual attendance, the system captures a student’s face through a camera, recognizes it using trained face encodings, and automatically records attendance with date and time in a CSV file.

                        
 ✨ Features

📸 Automatic face capture using webcam

🧠 Face recognition using trained face encodings

🕒 Real-time attendance marking with date & time

📄 Attendance stored in CSV format

🔁 Prevents duplicate attendance on the same day

🚀 Easy to run and user-friendly

🔐 No manual intervention required

                            
🛠 Technologies Used

Python 3.10

OpenCV – for image processing and camera access

NumPy – for numerical operations

face_recognition – for face detection and recognition

dlib – for face landmark detection

CSV Module – for attendance storage

              
                                 Folder Structure
                                                                                            
                                                                                            SMART_ATTENDANCE_GIT/
│
├── main.py               
├── face_capture.py
├── face_encode.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── dataset/
│   └── .gitkeep
│
├── attendance/
│   └── .gitkeep

                                 How to Run the Project                           

Capture student face images using webcam

Generate and store face encodings

Detect faces in real time

Match detected faces with stored encodings

Mark attendance automatically with timestamp

   Step 1: Install dependencies: pip install -r requirements.txt
   Step 2: Run the project :  python main.py

Use Cases : Colleges & Universities,Schools,office ,Labs & Training Centers

Future Enhancements :

GUI-based interface

Database integration

Cloud storage support

Mask detection support

Mobile app integration

