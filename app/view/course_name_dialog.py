from typing import List, Optional

from qfluentwidgets import BodyLabel, EditableComboBox, MessageBoxBase, SubtitleLabel


class CourseNameDialog(MessageBoxBase):
    """终了后填写课程名：可从已用过的课程名中选择，也可新输入；「スキップ」则不填写"""

    def __init__(self, course_names: List[str], parent=None):
        super().__init__(parent)
        self.titleLabel = SubtitleLabel("授業名を入力", self)
        self.hintLabel = BodyLabel("後から分析レポートで変更できます。", self)
        self.nameBox = EditableComboBox(self)
        self.nameBox.addItems(course_names)
        self.nameBox.setCurrentIndex(-1)
        self.nameBox.setPlaceholderText("例：線形代数 第3回")
        self.nameBox.setMinimumWidth(320)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.hintLabel)
        self.viewLayout.addWidget(self.nameBox)
        self.yesButton.setText("保存")
        self.cancelButton.setText("スキップ")

    def course_name(self) -> Optional[str]:
        return self.nameBox.text().strip() or None


class RenameCourseDialog(CourseNameDialog):
    """报告页「記録の管理」改课程名（#10）：预填当前课程名；清空后保存即改为未填写"""

    def __init__(self, course_names: List[str], current: Optional[str], parent=None):
        super().__init__(course_names, parent)
        self.titleLabel.setText("授業名を変更")
        self.hintLabel.setText("空欄で保存すると「授業名なし」になります。")
        self.nameBox.setText(current or "")
        self.cancelButton.setText("キャンセル")
