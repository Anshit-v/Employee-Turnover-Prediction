from flask import Flask, render_template, request
import pandas as pd
import pickle 

app = Flask(__name__)
model = pickle.load(open('rforest_model.pkl', 'rb'))

@app.route('/')

def home():
    #print("hlw1")    
    
    return render_template('code.html')
  
@app.route('/prediction',methods =['POST'])

def prediction():
    print("hlw2")    
    sales=0
    accounting=0
    hr=0
    technical =0
    support =0
    management =0
    it=0
    product_mng=0
    marketing = 0
    randD=0 

    if request.method == 'POST':
        """
        satisfaction_level = request.form['satisfaction_level']
        last_evaluation = request.form['last_evaluation']
        number_project = request.form['number_project']
        average_montly_hours = request.form['average_montly_hours']
        time_spend_company = request.form['time_spend_company']
        work_accident = request.form['work_accident']
        promotion_last_5years = request.form['promotion_last_5years']
        """
        department = request.form['department']
        
        
        randD = 0
        if department == "sales":
            sales = 1
        elif department == "accounting":
            accounting = 1
        elif department == "hr":
            hr = 1
        elif department == "technical":
            technical = 1
        elif department == "support":
            support = 1
        elif department == "management":
            management = 1
        elif department == "it":
            it = 1
        elif department == "product_mng":
            product_mng = 1
        elif department == "marketing":
            marketing = 1
        else:
            randD = 1
        
        final_data = pd.DataFrame({"Satisfaction_Level":[request.form['satisfaction_level']], "Last_Evaluation":[request.form['last_evaluation']], \
          "Number_Of_Projects":[request.form['number_project']], "Average_Monthly_Hours":[request.form['average_montly_hours']], "Time_Spend_company":[request.form['time_spend_company']],\
          "Work_Accident":[request.form['work_accident']], "Promotion_In_Last_5Years":[request.form['promotion_last_5years']], "Salary": [request.form['salary']], "department_IT":[it], \
          "department_RandD":[randD], "department_accounting":[accounting ],"department_hr":[hr], "department_management":[management], "department_marketing":[marketing], \
          "department_product_mng":[product_mng], "department_sales":[sales], "department_support":[support], "department_technical":[technical] })

        
        predi = model.predict(final_data)
        print(predi)        
           
        return render_template('code.html',prediction_text ="Prediction : {}".format("\n Employee might leave the organisation." if predi==1 else "\n Employee will stay in the organisation."))
    return render_template('code.html')
        
if __name__ == "__main__":
    app.run(debug=True)
