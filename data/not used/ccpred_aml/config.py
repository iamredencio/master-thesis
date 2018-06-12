import os
import pandas as pd
from datetime import date, timedelta

# directories
packagedir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(packagedir, 'data')

# team
team = 'Sparen & beleggen Totaal'

# dates
window = 180  # Trainingswindow
maxahead = 9  # Aantal dagen dat vooruit voorspeld wordt
periods = (window / 5 * 7) + 10 + 28 + maxahead + maxahead  # Benodigde dagen om model te schatten en uit te scoren
dates = pd.date_range(end=date.today() + timedelta(maxahead - 1), periods=periods)

