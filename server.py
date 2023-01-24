from flask import Flask, flash, render_template

app = Flask(__name__)
app.secret_key = 'mysecretkey'

@app.route('/')
def index():
    with app.app_context():
        flash('Bienvenido al servidor')
    return
render_template('index.html')

if __name__ == '__main__':
    app.run(port = 8000)


