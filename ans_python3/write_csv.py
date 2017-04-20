
import csv
history_fn = output_fn + '.csv'
with open(history_fn, 'w') as csv_file:
	w = csv.writer(csv_file)
	temp = numpy.array(list(fit_log.history.values()))
	w.writerow(list(fit_log.history.keys()))
	for i in range(temp.shape[1]):
		w.writerow(temp[:,i])
