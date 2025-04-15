import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go
import plotly.express as px
import sqlite3
import json
from datetime import datetime
from email.mime.text import MIMEText
import smtplib

# Initialize the app
def init():
    """Initialize the application"""
    st.set_page_config(page_title="Wind Turbine Monitoring", layout="wide")
    init_db()

# Database functions
def init_db():
    """Initialize the SQLite database"""
    conn = sqlite3.connect('turbine_monitoring.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (timestamp TEXT, turbine_id TEXT, failure_probability REAL, 
                  features TEXT, prediction_used INTEGER, notes TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (timestamp TEXT, turbine_id TEXT, alert_type TEXT, 
                  alert_message TEXT, resolved INTEGER)''')
    conn.commit()
    conn.close()

def log_prediction(turbine_id, failure_probability, features, notes=""):
    """Log a prediction to the database"""
    conn = sqlite3.connect('turbine_monitoring.db')
    c = conn.cursor()
    c.execute('''INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)''', 
              (datetime.now().isoformat(), turbine_id, failure_probability, 
               json.dumps(features), 0, notes))
    conn.commit()
    conn.close()

def log_alert(turbine_id, alert_type, alert_message):
    """Log an alert to the database"""
    conn = sqlite3.connect('turbine_monitoring.db')
    c = conn.cursor()
    c.execute('''INSERT INTO alerts VALUES (?, ?, ?, ?, ?)''', 
              (datetime.now().isoformat(), turbine_id, alert_type, 
               alert_message, 0))
    conn.commit()
    conn.close()

def get_recent_predictions(turbine_id=None, limit=100):
    """Get recent predictions from the database"""
    conn = sqlite3.connect('turbine_monitoring.db')
    query = 'SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?'
    params = (limit,)
    
    if turbine_id:
        query = 'SELECT * FROM predictions WHERE turbine_id = ? ORDER BY timestamp DESC LIMIT ?'
        params = (turbine_id, limit)
    
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Parse features JSON
        df['features_dict'] = df['features'].apply(json.loads)
        # Extract common features
        for col in ['rotor_speed', 'generator_speed', 'power_output', 'vibration', 'gearbox_temp']:
            df[col] = df['features_dict'].apply(lambda x: x.get(col, None))
    return df

# Alerting functions
def check_and_alert(turbine_id, failure_probability, features):
    """Check conditions and generate alerts"""
    alerts = []
    features_dict = features if isinstance(features, dict) else features.to_dict('records')[0]
    
    # Critical failure probability
    if failure_probability > 80:
        message = f"Critical failure risk ({failure_probability:.1f}%) detected"
        alerts.append(('CRITICAL', message))
        log_alert(turbine_id, 'CRITICAL', message)
        send_email_alert(turbine_id, 'CRITICAL', message, features_dict)
    
    # High failure probability
    elif failure_probability > 70:
        message = f"High failure risk ({failure_probability:.1f}%) detected"
        alerts.append(('HIGH', message))
        log_alert(turbine_id, 'HIGH', message)
    
    # Rapid increase detection
    recent = get_recent_predictions(turbine_id, limit=3)
    if len(recent) >= 3:
        trend = np.polyfit(range(3), recent['failure_probability'], 1)[0]
        if trend > 15:  # Rapid increase (>15% per reading)
            message = f"Rapid risk increase (trend: +{trend:.1f}% per reading)"
            alerts.append(('TREND', message))
            log_alert(turbine_id, 'TREND', message)
    
    # Sensor value thresholds
    if features_dict['vibration'] > 3.5:
        message = f"High vibration detected ({features_dict['vibration']:.2f} mm/s)"
        alerts.append(('VIBRATION', message))
        log_alert(turbine_id, 'VIBRATION', message)
    
    if features_dict['gearbox_temp'] > 90:
        message = f"High gearbox temperature ({features_dict['gearbox_temp']:.1f}°C)"
        alerts.append(('TEMPERATURE', message))
        log_alert(turbine_id, 'TEMPERATURE', message)
    
    return alerts

def send_email_alert(turbine_id, alert_type, message, features):
    """Send email alert (optional)"""
    # Configure these in your production environment
    email_enabled = False  # Set to True to enable email alerts
    if not email_enabled:
        return
    
    try:
        smtp_server = "your.smtp.server"
        smtp_port = 587
        smtp_user = "your@email.com"
        smtp_password = "yourpassword"
        to_emails = ["maintenance-team@yourcompany.com"]
        
        msg = MIMEText(
            f"Alert Details:\n\n"
            f"Turbine ID: {turbine_id}\n"
            f"Alert Type: {alert_type}\n"
            f"Message: {message}\n\n"
            f"Current Features:\n{json.dumps(features, indent=2)}"
        )
        
        msg['Subject'] = f"{alert_type} ALERT: {turbine_id}"
        msg['From'] = smtp_user
        msg['To'] = ", ".join(to_emails)
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, to_emails, msg.as_string())
        
        log_alert(turbine_id, 'EMAIL', f"Sent email alert: {message}")
    except Exception as e:
        log_alert('SYSTEM', 'EMAIL_ERROR', f"Failed to send email: {str(e)}")

# Visualization functions
def show_monitoring_dashboard():
    """Display the monitoring dashboard"""
    st.header("📊 Model Monitoring Dashboard")
    
    # Connect to database
    conn = sqlite3.connect('turbine_monitoring.db')
    
    # Load data
    pred_df = pd.read_sql('SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 500', conn)
    alert_df = pd.read_sql('SELECT * FROM alerts WHERE resolved = 0 ORDER BY timestamp DESC', conn)
    
    if not pred_df.empty:
        pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])
        # Parse features JSON
        pred_df['features_dict'] = pred_df['features'].apply(json.loads)
        # Extract common features
        for col in ['rotor_speed', 'generator_speed', 'power_output', 'vibration', 'gearbox_temp']:
            pred_df[col] = pred_df['features_dict'].apply(lambda x: x.get(col, None))
    
    if not alert_df.empty:
        alert_df['timestamp'] = pd.to_datetime(alert_df['timestamp'])
    
    # KPI Metrics
    st.subheader("📈 Key Performance Indicators")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Predictions", len(pred_df) if not pred_df.empty else 0)
    with cols[1]:
        high_risk = len(pred_df[pred_df['failure_probability'] > 70]) if not pred_df.empty else 0
        st.metric("High Risk Predictions", high_risk)
    with cols[2]:
        st.metric("Active Alerts", len(alert_df))
    with cols[3]:
        avg_prob = pred_df['failure_probability'].mean() if not pred_df.empty else 0
        st.metric("Avg Failure Probability", f"{avg_prob:.1f}%" if not pred_df.empty else "N/A")
    
    # Failure Probability Trends
    st.subheader("📉 Failure Probability Over Time")
    if not pred_df.empty:
        fig = px.line(pred_df, x='timestamp', y='failure_probability', 
                      color='turbine_id', title="Failure Probability Trend")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No prediction data available")
    
    # Feature Analysis
    st.subheader("🔍 Feature Analysis")
    if not pred_df.empty:
        feature = st.selectbox("Select feature to analyze", 
                             ['rotor_speed', 'generator_speed', 'power_output', 
                              'gearbox_temp', 'generator_temp', 'vibration'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"{feature} Over Time")
            fig = px.line(pred_df, x='timestamp', y=feature, color='turbine_id')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader(f"{feature} vs Failure Probability")
            fig = px.scatter(pred_df, x=feature, y='failure_probability', 
                            color='turbine_id', trendline="lowess")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No feature data available")
    
    # Active Alerts
    st.subheader("🚨 Active Alerts")
    if not alert_df.empty:
        for _, row in alert_df.iterrows():
            with st.expander(f"{row['timestamp']} - {row['turbine_id']}: {row['alert_type']}"):
                st.warning(row['alert_message'])
                if st.button(f"Mark as Resolved", key=f"resolve_{row['timestamp']}"):
                    c = conn.cursor()
                    c.execute('UPDATE alerts SET resolved = 1 WHERE timestamp = ?', 
                             (row['timestamp'].isoformat(),))
                    conn.commit()
                    st.rerun()
    else:
        st.success("No active alerts")
    
    conn.close()



# Prediction functions
def generate_live_data():
    """Generate simulated live turbine data"""
    live_data = {
        'Turbine_ID': np.random.choice([f'WTG{i+1}' for i in range(6)]),
        'rotor_speed': np.random.uniform(12.4, 16.3),
        'generator_speed': np.random.uniform(1430, 1661),
        'power_output': np.random.uniform(1535, 2515),
        'blade_pitch_angle': np.random.uniform(3.7, 7.2),
        'gearbox_temp': np.random.uniform(51.8, 94.4),
        'generator_temp': np.random.uniform(60.5, 111.8),
        'vibration': np.random.uniform(1.4, 3.3),
        'oil_pressure': np.random.uniform(92.2, 113.6),
        'wind_speed': np.random.uniform(4.6, 15.5),
        'ambient_temp': np.random.uniform(15.4, 27.0),
        'humidity': np.random.uniform(51.3, 67.5),
        'time_since_maintenance': np.random.randint(14, 330)
    }
    return pd.DataFrame(live_data, index=[0])

def get_user_input():
    """Get user input from sidebar"""
    st.sidebar.header("⚙️ Input Features")
    turbine_id = st.sidebar.selectbox('Turbine ID', [f'WTG{i+1}' for i in range(10)])
    rotor_speed = st.sidebar.slider('Rotor Speed (RPM)', 10.0, 20.0, 15.0)
    generator_speed = st.sidebar.slider('Generator Speed (RPM)', 1400.0, 1600.0, 1500.0)
    power_output = st.sidebar.slider('Power Output (kW)', 1500.0, 2500.0, 2000.0)
    blade_pitch_angle = st.sidebar.slider('Blade Pitch Angle (degrees)', 0.0, 10.0, 5.0)
    gearbox_temp = st.sidebar.slider('Gearbox Temperature (°C)', 50.0, 100.0, 70.0)
    generator_temp = st.sidebar.slider('Generator Temperature (°C)', 60.0, 110.0, 80.0)
    vibration = st.sidebar.slider('Vibration (mm/s)', 1.0, 5.0, 2.0)
    oil_pressure = st.sidebar.slider('Oil Pressure (bar)', 80.0, 120.0, 100.0)
    wind_speed = st.sidebar.slider('Wind Speed (m/s)', 5.0, 15.0, 10.0)
    ambient_temp = st.sidebar.slider('Ambient Temperature (°C)', 10.0, 30.0, 20.0)
    humidity = st.sidebar.slider('Humidity (%)', 40.0, 80.0, 60.0)
    time_since_maintenance = st.sidebar.slider('Time Since Last Maintenance (days)', 0, 365, 180)

    return {
        'Turbine_ID': turbine_id,
        'rotor_speed': rotor_speed,
        'generator_speed': generator_speed,
        'power_output': power_output,
        'blade_pitch_angle': blade_pitch_angle,
        'gearbox_temp': gearbox_temp,
        'generator_temp': generator_temp,
        'vibration': vibration,
        'oil_pressure': oil_pressure,
        'wind_speed': wind_speed,
        'ambient_temp': ambient_temp,
        'humidity': humidity,
        'time_since_maintenance': time_since_maintenance
    }

# Main application
def main():
    """Main application function"""
    init()
    
    # Load the model
    try:
        model = joblib.load('best_model.pkl')
    except:
        st.error("Failed to load the predictive model. Please ensure 'best_model.pkl' exists.")
        return
    
    # Sidebar navigation
    st.sidebar.header("Navigation")
    app_mode = st.sidebar.radio("Select Mode", 
                               ["📊 Monitoring Dashboard", 
                                "🖥️ Interactive Prediction", 
                                "📁 Batch Prediction", 
                                "🌪️ Live Simulation"])
    
    # Main content area
    st.title("🌬️ Wind Turbine Predictive Maintenance System")
    
    if app_mode == "📊 Monitoring Dashboard":
        show_monitoring_dashboard()
    
    elif app_mode == "🖥️ Interactive Prediction":
        st.header("Interactive Prediction")
        user_data = get_user_input()
        features = pd.DataFrame(user_data, index=[0])
        
        st.subheader("Input Features")
        st.write(features)
        
        # Make prediction
        prediction = model.predict_proba(features.drop(columns=['Turbine_ID']))[:, 1][0]
        failure_probability = round(prediction * 100, 2)
        failure_probability = min(max(failure_probability, 0), 100)
        
        # Display results
        st.subheader("Prediction Results")
        st.metric("Failure Probability", f"{failure_probability}%")
        
        if prediction > 0.7:
            st.error("🚨 Warning: High probability of failure detected!")
        elif prediction > 0.5:
            st.warning("⚠️ Caution: Moderate probability of failure detected")
        else:
            st.success("✅ Normal operation")
        
        # Log the prediction
        log_prediction(user_data['Turbine_ID'], failure_probability, 
                      features.drop(columns=['Turbine_ID']).to_dict('records')[0])
        
        # Check for alerts
        alerts = check_and_alert(user_data['Turbine_ID'], failure_probability, 
                               features.drop(columns=['Turbine_ID']))
        for alert_type, message in alerts:
            if alert_type == 'CRITICAL':
                st.error(f"ALERT: {message}")
            else:
                st.warning(f"Warning: {message}")
    
    elif app_mode == "📁 Batch Prediction":
        st.header("Batch Prediction")
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                required_cols = ['Turbine_ID', 'rotor_speed', 'generator_speed', 'power_output',
                               'blade_pitch_angle', 'gearbox_temp', 'generator_temp', 'vibration',
                               'oil_pressure', 'wind_speed', 'ambient_temp', 'humidity',
                               'time_since_maintenance']
                
                if all(col in df.columns for col in required_cols):
                    # Make predictions
                    df['Failure_Probability'] = model.predict_proba(df[required_cols[1:]])[:, 1] * 100
                    df['Alert'] = df['Failure_Probability'].apply(
                        lambda x: 'CRITICAL' if x > 80 else 'HIGH' if x > 70 else 'WARNING' if x > 50 else 'NORMAL')
                    
                    # Show results
                    st.subheader("Prediction Results")
                    st.dataframe(df)
                    
                    # Download button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Predictions",
                        csv,
                        "turbine_predictions.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
                    # Log predictions
                    for _, row in df.iterrows():
                        features = {col: row[col] for col in required_cols[1:]}
                        log_prediction(row['Turbine_ID'], row['Failure_Probability'], features, 
                                      f"Batch upload - {uploaded_file.name}")
                else:
                    st.error(f"Missing required columns. Needed: {required_cols}")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    elif app_mode == "🌪️ Live Simulation":
        st.header("Live Turbine Monitoring Simulation")
        
        # Initialize session state for simulation control
        if 'simulation_running' not in st.session_state:
            st.session_state.simulation_running = False
        
        # Start/Stop buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Simulation", key="start_simulation"):
                st.session_state.simulation_running = True
        with col2:
            if st.button("Stop Simulation", key="stop_simulation"):
                st.session_state.simulation_running = False
        
        if st.session_state.simulation_running:
            st.info("Simulating live turbine data... Press Stop to end")
            
            # Create placeholders only once outside the loop
            live_data_placeholder = st.empty()
            alert_placeholder = st.empty()
            chart_placeholder = st.empty()
            
            # Initialize chart
            fig = go.Figure()
            fig.update_layout(
                title="Live Turbine Metrics",
                xaxis_title="Time",
                yaxis_title="Value",
                height=500
            )
            
            # Store data for the chart
            if 'chart_data' not in st.session_state:
                st.session_state.chart_data = {
                    'time': [],
                    'power_output': [],
                    'failure_prob': []
                }
            
            # Simulation loop
            while st.session_state.simulation_running:
                # Generate new data
                live_data = generate_live_data()
                
                # Make prediction
                prediction = model.predict_proba(live_data.drop(columns=['Turbine_ID']))[:, 1][0]
                failure_probability = round(prediction * 100, 2)
                failure_probability = min(max(failure_probability, 0), 100)
                
                # Log data
                log_prediction(live_data['Turbine_ID'].values[0], failure_probability, 
                            live_data.drop(columns=['Turbine_ID']).to_dict('records')[0])
                
                # Check for alerts
                alerts = check_and_alert(live_data['Turbine_ID'].values[0], failure_probability, 
                                    live_data.drop(columns=['Turbine_ID']))
                
                # Display data
                with live_data_placeholder.container():
                    st.subheader("Current Turbine Status")
                    st.dataframe(live_data)
                    
                    # Display alerts
                    if alerts:
                        for alert_type, message in alerts:
                            if alert_type == 'CRITICAL':
                                st.error(f"🚨 {message}")
                            else:
                                st.warning(f"⚠️ {message}")
                    else:
                        st.success("✅ Normal operation")
                
                # Update chart data
                current_time = datetime.now()
                st.session_state.chart_data['time'].append(current_time)
                st.session_state.chart_data['power_output'].append(live_data['power_output'].values[0])
                st.session_state.chart_data['failure_prob'].append(failure_probability)
                
                # Update chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.chart_data['time'],
                    y=st.session_state.chart_data['power_output'],
                    name="Power Output",
                    mode='lines+markers'
                ))
                fig.add_trace(go.Scatter(
                    x=st.session_state.chart_data['time'],
                    y=st.session_state.chart_data['failure_prob'],
                    name="Failure Probability",
                    yaxis='y2',
                    mode='lines+markers'
                ))
                fig.update_layout(yaxis2=dict(title='Failure Probability (%)', overlaying='y', side='right'))
                
                with chart_placeholder.container():
                    st.plotly_chart(fig, use_container_width=True)
                
                # Add small delay
                time.sleep(2)
                
                # Check if we should still be running
                if not st.session_state.simulation_running:
                    break
        
        elif app_mode == "🌪️ Live Simulation":
            # st.header("Live Turbine Monitoring Simulation")
            
            # if st.button("Start Simulation"):
            #     st.info("Simulating live turbine data... Press Stop to end")
            #     stop_button_key = "stop_simulation"  # Unique key for the stop button
            #     live_data_placeholder = st.empty()
            #     alert_placeholder = st.empty()
            #     chart_placeholder = st.empty()

                # # Initialize chart
                # fig = go.Figure()
                # fig.update_layout(
                #     title="Live Turbine Metrics",
                #     xaxis_title="Time",
                #     yaxis_title="Value",
                #     height=500
                # )

                # while True:
                #     # Generate new data
                #     live_data = generate_live_data()

                #     # Make prediction
                #     prediction = model.predict_proba(live_data.drop(columns=['Turbine_ID']))[:, 1][0]
                #     failure_probability = round(prediction * 100, 2)
                #     failure_probability = min(max(failure_probability, 0), 100)

                #     # Log data
                #     log_prediction(live_data['Turbine_ID'].values[0], failure_probability, 
                #                 live_data.drop(columns=['Turbine_ID']).to_dict('records')[0])

                #     # Check for alerts
                #     alerts = check_and_alert(live_data['Turbine_ID'].values[0], failure_probability, 
                #                             live_data.drop(columns=['Turbine_ID']))

                #     # Display data
                #     live_data_placeholder.subheader("Current Turbine Status")
                #     live_data_placeholder.dataframe(live_data)

                #     # Display alerts
                #     if alerts:
                #         alert_placeholder.empty()
                #         for alert_type, message in alerts:
                #             if alert_type == 'CRITICAL':
                #                 alert_placeholder.error(f"🚨 {message}")
                #             else:
                #                 alert_placeholder.warning(f"⚠️ {message}")

                    # # Update chart
                    # fig.add_trace(go.Scatter(
                    #     x=[datetime.now()],
                    #     y=[live_data['power_output'].values[0]],
                    #     name="Power Output",
                    #     mode='lines+markers'
                    # ))
                    # fig.add_trace(go.Scatter(
                    #     x=[datetime.now()],
                    #     y=[failure_probability],
                    #     name="Failure Probability",
                    #     yaxis='y2',
                    #     mode='lines+markers'
                    # ))
                    # fig.update_layout(yaxis2=dict(title='Failure Probability (%)', overlaying='y', side='right'))
                    # chart_placeholder.plotly_chart(fig, use_container_width=True)

                    # # Check if stop button was pressed
                    # if st.button("Stop Simulation", key=stop_button_key):
                    #     break

                time.sleep(2)  # Simulate 2-second delay between readings

if __name__ == "__main__":
    main()