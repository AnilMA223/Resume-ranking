from flask import *  
import pickle
import docx2txt
import PyPDF2 
import docx
from docx import Document
import re
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename
import urllib.request
import pandas as pd
from IPython.display import HTML
import glob
import spacy
from spacy.matcher import Matcher
from pdf2docx import Converter
import shutil
import plotly
import plotly.express as px


app = Flask(__name__)
  
UPLOAD_PATH=r'saved files\resume_files'
UPLOAD_PATH1=r'saved files\job_description_file'
PDF_PATH=r'saved files\pdf_files'
CSV_PATH=r'saved files\csv_file'
app.config["UPLOAD_PATH"]=UPLOAD_PATH
app.config["UPLOAD_PATH1"]=UPLOAD_PATH1
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
mypath = UPLOAD_PATH
path_output =UPLOAD_PATH

 
@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")  
 
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        for f in request.files.getlist('resume'):
         f.save(os.path.join(app.config['UPLOAD_PATH'], f.filename))
        for f1 in request.files.getlist('jd'):
         f1.save(os.path.join(app.config['UPLOAD_PATH1'], f1.filename))
    
       
        
        jd=docx2txt.process(f1)
    path =UPLOAD_PATH
    path1=PDF_PATH
    
            
    candidate_resume=[]
    for files in os.listdir(path):
        candidate_resume.append(files)
    for files in os.listdir(path):
        if files.lower().endswith(".pdf"):
            shutil.move(os.path.join(path, files), path1)
    path_input = PDF_PATH + '\\'
    path_output = UPLOAD_PATH + '\\'
    for file in os.listdir(path_input):
        cv = Converter(path_input+file)
        cv.convert(path_output+file[:-4]+'.docx', start=0, end=None)
        cv.close()
    num=1
    res=[]
    res_files=[]
    for file in os.scandir(path):
        r=docx2txt.process(file)
        num=num+1
        res_files.append(r)
        text=[r,jd]
        cv=CountVectorizer()
        count_matrix=cv.fit_transform(text)
        matchpercentage=round(cosine_similarity(count_matrix)[0][1]*100,2)
        res.append(matchpercentage)
    
    
    email=[]
    for x1 in res_files:
        emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", x1)
        email.append(emails)
    
    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)
    
        
    def extract_name(resume_text):
        nlp_text = nlp(resume_text)   
        pattern = [[{'POS': 'PROPN'}, {'POS': 'PROPN'}]]
        matcher.add(resume_text,pattern) 
        matches = matcher(nlp_text)
   
        for match_id, start, end in matches:
            span = nlp_text[start:end]
            return span.text
    names_=[]
    for u in res_files: 
        ex_text=extract_name(u)
        names_.append(ex_text)
    
    data = pd.DataFrame({'Resume File Name' : candidate_resume, 'Matching %' : res,'Email List':email,'Name List':names_}).sort_values(['Matching %'],ascending=False).reset_index(drop=True)
    
    headings=data.columns
    data_values=data.values
    #data.to_csv(CSV_PATH + '\\data.csv')
    
    nlp = spacy.load('en_core_web_sm')
    def extract_skills(resume_text):
        nlp_text = nlp(resume_text)
        noun_chunks = nlp_text.noun_chunks


    # removing stop words and implementing word tokenization
        tokens = [token.text for token in nlp_text if not token.is_stop]
    
    # reading the csv file
        data = pd.read_csv(r"Skills File\jd skills.csv") 
        print(data.columns.values[0].split(','))
    # extract values
        skills =data.columns.values[0].split(',')
    #print(skills)
        skillset = []
    
    # check for one-grams (example: python)
        for token in tokens:
            if token.lower() in skills:
                skillset.append(token)
    
    # check for bi-grams and tri-grams (example: machine learning)
        for token in noun_chunks:
            token = token.text.lower().strip()
            if token in skills:
                skillset.append(token)
    
    
    
        return [i.capitalize() for i in set([i.lower() for i in skillset])]
    
    Skill=[]
    for k in res_files:
        ex_skills=extract_skills(k)
        #ex_skills=str( ex_skills)
        Skill.append(ex_skills)
        
    data1=data = pd.DataFrame({'Resume File Name' : candidate_resume, 'Matching %' : res,'Email List':email,'Name List':names_,'Skills':Skill}).reset_index(drop=True)
    data1.to_csv(CSV_PATH + '\\analytics.csv')
    df = pd.read_csv(CSV_PATH + '\\analytics.csv')
    df.rename({'Unnamed: 0':'rank'},axis=1,inplace=True) 
    Rank=[]
    for i in df['rank']:
        i=i+1
        Rank.append(i)
    df['Rank']=Rank
    df.drop('rank',axis=1)
    newcolumns=['Rank','Resume File Name', 'Matching %', 'Email List', 'Name List','Skills']
    Analytic=df[newcolumns]
    
    #import plotly.graph_objects as go
    import plotly.express as px
    Analytic = Analytic.sort_values('Matching %', ascending=False)
    fig = px.bar(Analytic, x="Matching %", y="Name List",text="Skills", color="Skills",title="Analytics")
    fig.update_traces(textposition='auto')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
  

   
    files = os.listdir(UPLOAD_PATH)
    for f1 in files:
        os.remove(UPLOAD_PATH +"\\"+ f1)
    for files in os.listdir(path1):
        os.remove(path1 +"\\"+ files)

    return render_template('analytics.html', graphJSON=graphJSON,headings=headings,data=data_values)
    #return render_template('datatohtml.html',)

if __name__ == '__main__':  
    app.run(debug = True)  