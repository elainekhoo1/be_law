# importing the required libraries
from flask import Flask, render_template, request, redirect, url_for
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from joblib import load
import logging
import os
os.chdir(r'D:\00 Self-Learnings\0 Learning and Wellbeing (LAW)\be_law')

logging.basicConfig(filename='./data/api_logs/job_description_classification.log', level=logging.DEBUG,
                    format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')


# load the pipeline object
pipeline = load("./src/models/job_description_classification_lgbm.joblib")

# function to get results for a particular text query
def requestResults(name):
    # get the tweets text
    # tweets = get_related_tweets(name)
    inputs = [name]
    return pipeline.predict(inputs)

def create_app():

    # start flask
    app = Flask(__name__)
    app.config["DEBUG"] = True

    # render default webpage
    @app.route('/')
    def home():
        user_ip = request.remote_addr
        app.logger.info(f"[{user_ip}] Default home '/' is called.")
        return render_template('landing.html')

    @app.route('/info/')
    def information():
        return render_template('display_plots.html', name='Sample Plot', url='/static/images/clustered_results_vs_manual_tags.png')

    @app.route('/post-opt-nb/')
    def post_opt_nb_jupyter():
        return render_template('post_opt_nb.html')

    # when the post method detect, then redirect to success function
    @app.route('/', methods=['POST', 'GET'])
    def get_data():
        if request.method == 'POST':
            user = request.form['jd_search']
            return redirect(url_for('success', name=user))

    # get the data for the requested query
    @app.route('/success/<name>')
    def success(name):
        return "<xmp>" + str(requestResults(name)) + " </xmp> "

    return app


if __name__ == '__main__':

    app = create_app()
    app.run(host='0.0.0.0', port=8888, debug=True, use_reloader=False, threaded=True)


