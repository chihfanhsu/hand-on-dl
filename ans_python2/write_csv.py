import csv
history_fn = output_fn + '.csv'
with open(history_fn, 'wb') as csv_file:
	w = csv.writer(csv_file)
	temp = numpy.array(fit_log.history.values())
	w.writerow(fit_log.history.keys())
	for i in range(temp.shape[1]):
		w.writerow(temp[:,i])