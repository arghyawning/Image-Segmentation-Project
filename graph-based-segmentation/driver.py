import sys
from PyQt5.QtWidgets import QApplication

from GUI import GUI
from Graph import Graph


def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    graph = Graph()
    ui = GUI(graph)
    ui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
