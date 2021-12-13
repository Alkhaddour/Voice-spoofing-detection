# Voice spoofing detection
<p>In this project I created a voice spoofing detection system using LSTM. <br> </p>

### How to run the project
<ol>
<li>You can download the data through <a href="https://mfd.sk/nX4LvUe9xl3k5XmbciY1nwUq"> this link</a> </li>
<li>Extract the data and update paths in <a href="https://github.com/Alkhaddour/Voice-spoofing-detection/blob/main/config.py"> config.py</a> file.</li>
<li>Extract features using <a href="https://github.com/Alkhaddour/Voice-spoofing-detection/blob/main/config.py"> Data Extraction.ipynb notebook.</a> </li>
<li>Train the model using  <a href="https://github.com/Alkhaddour/Voice-spoofing-detection/blob/main/config.py"> train_model.ipynb notebook.</a> </li>
<li>Validate the model by maximizing AUPRC using <a href="https://github.com/Alkhaddour/Voice-spoofing-detection/blob/main/validate_AUPRC.ipynb"> validate_AUPRC.ipynb notebook.</a> </li>
  <li>Validate the model by minimizing EER using <a href="https://github.com/Alkhaddour/Voice-spoofing-detection/blob/main/validate_EER.ipynb"> validate_EER.ipynb notebook.</a> </li>
</ol>

### Results
<p>
  The models achieved 98.1% train accuracy and 97.4% validation accuracy while maximizing AUPRC, and  97.2% train accuracy and 96.4% validation accuracy while minimizing EER. Full list of metrics available <a href="https://github.com/Alkhaddour/Voice-spoofing-detection/blob/main/output/metrics.csv"> here</a> and <a href="https://github.com/Alkhaddour/Voice-spoofing-detection/blob/main/output/metrics_EER.csv"> here</a>.
</p>
