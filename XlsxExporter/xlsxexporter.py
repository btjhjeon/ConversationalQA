import xlsxwriter
import time

class XlsxExporter(object):
	def __init__(self, story, filepath=None):
		if filepath is None:
			self._filepath = './results/' + story + time.strftime("_%Y%m%d%H%M" ,time.localtime()) + '.xlsx'
		else:
			self._filepath = filepath

		self._data = []
		self._current_row = 0
		self._accuracy = None
		self._question = None
		self._desire = None
		self._pred = None


	def set_header(self, dic):
		self._header = dic


	def add_result(self, result):
		self._data.append(result)


	def set_answers(self, question, desire, pred):
		self._question = question
		self._desire = desire
		self._pred = pred


	def set_accuracy(self, acc):
		self._accuracy = acc


	def make_file(self):
		workbook = xlsxwriter.Workbook(self._filepath)
		data_worksheet = workbook.add_worksheet()
		answer_worksheet = workbook.add_worksheet()
		chart1 = workbook.add_chart({'type': 'line'})

		self._write_header(data_worksheet)
		self._write_results(data_worksheet)
		self._write_accuracy(data_worksheet)
		self._write_answers(answer_worksheet)

		workbook.close()


	def _write_header(self, worksheet):
		worksheet.write_row('A1', self._header.keys())
		worksheet.write_row('A2', self._header.values())
		self._current_row = 2


	def _write_results(self, worksheet):
		self._current_row += 1

		elements = self._data[0].keys()
		# worksheet.write_row(self._current_row, 1, range(1, len(self._data)+1))
		worksheet.write_row(self._current_row, 1, elements)
		row = self._current_row

		for data in self._data:
			self._current_row += 1
			column = 0
			for element in elements:
				column += 1
				worksheet.write(self._current_row, column, data[element])

		self._current_row += 1


	def _write_accuracy(self, worksheet):
		if self._accuracy is not None:
			self._current_row += 1
			worksheet.write(self._current_row, 0, 'Testing Accuracy')
			worksheet.write(self._current_row, 1, self._accuracy)


	def _write_answers(self, worksheet):
		if self._question is not None and self._desire is not None and self._pred is not None:
			worksheet.write('A1', 'Question')
			worksheet.write('B1', 'Desire')
			worksheet.write('C1', 'Prediction')
			worksheet.write_column('A2', self._question)
			worksheet.write_column('B2', self._desire)
			worksheet.write_column('C2', self._pred)
