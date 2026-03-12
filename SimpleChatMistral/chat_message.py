
class ChatMessage:
	def __init__(self, role, content):
		self.role = role
		self.content = content

	def format(self):
		return {"role":self.role, "content":self.content}