# claims-dashboard
A Dash/Plotly dashboard for visualizing large (re)insurance claims with the ability to fit severity and frequency distributions.

#reinsurance #actuary

Language: Python, I used version 3.7.9 to develop the app.

Modules:
<ul>
  <li>Dash 1.20.0</li>
  <li>Plotly 5.2.1</li>
  <li>Openpyxl 3.0.7</li>
  <li>Pandas 1.2.4</li>
  <li>Scipy 1.7.0</li>
 </ul>

Installation:
For a local desktop installation, only app.py is really needed from my github. You may download requirements.txt in order to:
<br>C:\\..\Python37\python.exe -m pip install -r requirements.txt (on Windows and similarly on other operating systems).
<br>In case "ERROR: Failed building wheel for bottleneck" shows up when installing the modules, you will need to install bottleneck manually (I used version 1.3.2).

Source of data: my imagination.

<img src='https://raw.githubusercontent.com/elsaburren/claims-dashboard/main/images/claims_dashboard.png' alt='claims dashboard preview'>

