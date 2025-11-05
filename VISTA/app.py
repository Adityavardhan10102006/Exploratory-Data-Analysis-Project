from flask import Flask, render_template, request, redirect, session
import mysql.connector

app = Flask(__name__)
app.secret_key = "secret123"

db = mysql.connector.connect(host="localhost", user="root", password="", database="student_portal")
cursor = db.cursor(dictionary=True)

@app.route('/')
def home():
    return redirect('/login')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        cursor.execute("SELECT * FROM Users WHERE username=%s AND password=%s", (uname, pwd))
        user = cursor.fetchone()
        if user:
            session['user'] = user
            return redirect('/dashboard')
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/login')
    return render_template('dashboard.html', user=session['user'])

@app.route('/students')
def students():
    cursor.execute("SELECT * FROM Student")
    students = cursor.fetchall()
    return render_template('students.html', students=students)

@app.route('/add_student', methods=['POST','GET'])
def add_student():
    if request.method == 'POST':
        name = request.form['name']
        dob = request.form['dob']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        course = request.form['course']
        cursor.execute("INSERT INTO Student(name,dob,email,phone,address,course) VALUES(%s,%s,%s,%s,%s,%s)", 
                       (name,dob,email,phone,address,course))
        db.commit()
        return redirect('/students')
    return render_template('add_student.html')

if __name__ == "__main__":
    app.run(debug=True)
