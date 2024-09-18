from fastapi import FastAPI, HTTPException, Form,UploadFile,File
import pickle
from datetime import date, datetime, timedelta
import pandas as pd
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
# Load the model from the file
with open('start_date_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the model from the file
with open('end_date_model.pkl', 'rb') as file:
    end_date_model = pickle.load(file)


def predict_next_start_date(period_start_date):
    # Convert date to ordinal
    period_start_date_ordinal = period_start_date.toordinal()
    # Create a DataFrame for the input
    input_df = pd.DataFrame([[period_start_date_ordinal]], columns=['Date_Ordinal'])
    # Predict the next date
    next_start_date_ordinal = model.predict(input_df)[0]
    # Convert ordinal back to date
    next_start_date = date.fromordinal(int(next_start_date_ordinal))
    return next_start_date

def predict_end_date(year, month, day, model):
    # Create a DataFrame from input date
    new_start_data = {
        'Day': [day],
        'Month': [month],
        'Year': [year]
    }

    new_start_df_end = pd.DataFrame(new_start_data)
    
    # Scale the new data
    # new_start_scaled = scaler.transform(new_start_df_end)
    
    # Make predictions
    pred_duration = model.predict(new_start_df_end)
    
    # Convert the predicted duration back to a date
    new_start_date = datetime(year, month, day)
    pred_end_date = new_start_date + pd.Timedelta(days=pred_duration[0])
    
    return pred_end_date.date()

def calculate_phases_dynamic(next_start_date, period_end_predicted_date, average_cycle, menstruation):
    if average_cycle < 22 or menstruation < 2 or menstruation >= average_cycle:
        return "Invalid input. Please enter valid average cycle and menstruation duration."
    
    # Ovulation day is typically 14 days before the end of the cycle
    ovulation_date = next_start_date - timedelta(days=14)

    # Follicular phase starts the day after menstruation ends
    follicular_start = period_end_predicted_date + timedelta(days=1)
    follicular_end = ovulation_date - timedelta(days=4)  # assuming follicular phase ends 4 days before ovulation
    
    if follicular_end < follicular_start:
        return "Invalid calculation based on provided inputs. Please adjust the input values."

    # follicular = [follicular_start.strftime('%Y-%m-%d'), follicular_end.strftime('%Y-%m-%d')]
    follicular = {'start':follicular_start.strftime('%Y-%m-%d'), 'end':follicular_end.strftime('%Y-%m-%d')}

    # Fertility window is typically 5 days, ending on the ovulation day
    fertility_start = ovulation_date - timedelta(days=3)
    fertility_end = ovulation_date + timedelta(days=2)
    # fertility = [fertility_start.strftime('%Y-%m-%d'), fertility_end.strftime('%Y-%m-%d')]
    fertility = {'start':fertility_start.strftime('%Y-%m-%d'), 'end':fertility_end.strftime('%Y-%m-%d')}

    # Luteal phase starts the day after ovulation and lasts until the end of the cycle
    luteal_start = ovulation_date + timedelta(days=3)
    luteal_end = next_start_date - timedelta(days=1)
    # luteal = [luteal_start.strftime('%Y-%m-%d'), luteal_end.strftime('%Y-%m-%d')]
    luteal = {'start':luteal_start.strftime('%Y-%m-%d'), 'end':luteal_end.strftime('%Y-%m-%d')}
    

    return {
        "follicular": follicular,
        "fertility": fertility,
        "ovulation": ovulation_date.strftime('%Y-%m-%d'),
        "luteal": luteal
    }

app = FastAPI()

def get_prediction(period_start_date):
    cycle_phases = {}
    if period_start_date:
        # period_start_date = date(int(period_start_date[0]), int(period_start_date[1]), int(period_start_date[2]))
        # period_start_date = date(period_start_date[0], period_start_date[1], period_start_date[2])

        period_start_date = date(period_start_date[2], period_start_date[1], period_start_date[0])

        print('\nCurrent period start date: ', period_start_date)
        period_end_predicted_date = predict_end_date(period_start_date.year, period_start_date.month, period_start_date.day, end_date_model)
        # print(period_end_predicted_date)
        print('\nPeriod end date: ', period_end_predicted_date)
        period_duration = period_end_predicted_date - period_start_date
        print('\nPeriod duration:', period_duration)
        next_start_pred_value = predict_next_start_date(period_start_date)
        cycle_duration = next_start_pred_value - period_start_date
        print('\nDuration in this cycle: ', cycle_duration)
        print('\nNext period start date: ', next_start_pred_value)


        period = {'start': period_start_date.strftime('%Y-%m-%d'), 'end': period_end_predicted_date.strftime('%Y-%m-%d')}

        cycle_phases['periods'] = period
        
        remain_phases = calculate_phases_dynamic(next_start_pred_value, period_end_predicted_date, cycle_duration.days, period_duration.days)
        
        cycle_phases.update(remain_phases)

        cycle_phases['no_of_days'] = cycle_duration.days

        cycle_phases['next_period_start_date'] = next_start_pred_value

        
        # phases['asa'] = 'asas'
        print('\n',cycle_phases)
        return cycle_phases
    else:
        # return 'no prediction made'
        return []

    

@app.post("/cycle_predictions/")
async def menst(period_start_date: str = Form(...)):
    period_start_date = period_start_date.split('/')
    period_start_date = [int(element) for element in period_start_date]
    print('************* period_start_date', period_start_date)
    cycle_prediction = get_prediction(period_start_date)
    if cycle_prediction:
        return {"cycle_predictions": cycle_prediction}
    else:
        raise HTTPException(status_code=404, detail="No recommendations found.")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=3535)