from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QLabel
from qfluentwidgets import (
    CardWidget, TitleLabel, SubtitleLabel, BodyLabel, 
    IconWidget, FluentIcon as FIF, ScrollArea
)

class StatCard(CardWidget):
    """
    A reusable helper widget for displaying a single statistic.
    Now just test
    """
    def __init__(self, icon, title, value, parent=None):
        super().__init__(parent)
        self.hLayout = QHBoxLayout(self)
        
        self.iconWidget = IconWidget(icon, self)
        self.iconWidget.setFixedSize(48, 48)
        
        self.textLayout = QVBoxLayout()
        self.valueLabel = TitleLabel(value, self)
        self.titleLabel = BodyLabel(title, self)
        self.textLayout.addWidget(self.valueLabel)
        self.textLayout.addWidget(self.titleLabel)
        
        self.hLayout.addWidget(self.iconWidget)
        self.hLayout.addLayout(self.textLayout)
        self.hLayout.addStretch(1)
        self.hLayout.setContentsMargins(20, 15, 20, 15)

class ReportInterface(ScrollArea):
    """
    The analysis and reporting page.
    Displays session statistics and charts.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.view = QWidget(self)
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.setObjectName("report-interface")
        
        # --- Layouts ---
        self.vBoxLayout = QVBoxLayout(self.view)
        self.statsGridLayout = QGridLayout()
        
        # --- UI Components ---
        
        self.titleLabel = TitleLabel("Session Analysis", self.view)
        self.subtitleLabel = BodyLabel("Review attention metrics and performance history.", self.view)
        
        # 1. Summary Cards
        self.cardTotalTime = StatCard(FIF.HISTORY, "Total Focus Time", "0h 00m", self.view)
        self.cardAvgScore = StatCard(FIF.HEART, "Avg Attention Score", "0%", self.view)
        self.cardDistractions = StatCard(FIF.REMOVE, "Distraction Events", "0", self.view)
        self.cardPeakTime = StatCard(FIF.HIGHTLIGHT, "Peak Focus Hour", "--:--", self.view)

        # 2. Chart Placeholder
        # In a real app, you would embed a Matplotlib canvas or PyQtGraph widget here.
        self.chartContainer = CardWidget(self.view)
        self.chartLayout = QVBoxLayout(self.chartContainer)
        self.chartTitle = SubtitleLabel("Attention Over Time", self.chartContainer)
        
        # Placeholder frame for the chart
        self.chartFrame = QFrame(self.chartContainer)
        self.chartFrame.setStyleSheet("background-color: rgba(0, 0, 0, 0.05); border-radius: 8px; border: 1px dashed #808080;")
        self.chartFrame.setMinimumHeight(400)
        self.chartLabel = QLabel("Chart Visualization Placeholder (Matplotlib / PyQtGraph)", self.chartFrame)
        self.chartLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        chartFrameLayout = QVBoxLayout(self.chartFrame)
        chartFrameLayout.addWidget(self.chartLabel)
        
        self.chartLayout.addWidget(self.chartTitle)
        self.chartLayout.addSpacing(10)
        self.chartLayout.addWidget(self.chartFrame)
        self.chartLayout.setContentsMargins(20, 20, 20, 20)

        # --- Assembly ---
        
        self.vBoxLayout.addWidget(self.titleLabel)
        self.vBoxLayout.addWidget(self.subtitleLabel)
        self.vBoxLayout.addSpacing(20)
        
        # Add cards to grid
        self.statsGridLayout.addWidget(self.cardTotalTime, 0, 0)
        self.statsGridLayout.addWidget(self.cardAvgScore, 0, 1)
        self.statsGridLayout.addWidget(self.cardDistractions, 1, 0)
        self.statsGridLayout.addWidget(self.cardPeakTime, 1, 1)
        self.statsGridLayout.setSpacing(20)
        
        self.vBoxLayout.addLayout(self.statsGridLayout)
        self.vBoxLayout.addSpacing(20)
        self.vBoxLayout.addWidget(self.chartContainer)
        self.vBoxLayout.addStretch(1)
        
        # Style
        self.vBoxLayout.setContentsMargins(30, 30, 30, 30)
        self.setStyleSheet("background-color: transparent;") # Let window background show

    # --- Placeholder Backend Methods ---

    def load_session_data(self, data: dict):
        """
        Populate the interface with data from a completed session.
        """
        print(f"Backend Hook: Loading data {data}")
        # Example update:
        # self.cardTotalTime.valueLabel.setText(data.get("duration", "0h 00m"))
        # self.cardAvgScore.valueLabel.setText(f"{data.get('avg_score', 0)}%")
        pass

    def render_charts(self):
        """
        Draw the graphs using the loaded data.
        """
        print("Backend Hook: Initializing Matplotlib figure here.")
        pass
