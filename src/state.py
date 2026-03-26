class AppState:
    def __init__(self):
        self.reset_all()

    def reset_all(self):
        self.chunks: list[str] = []
        self.index = None
        self.bm25 = None
        self.pdf_loaded: bool = False
