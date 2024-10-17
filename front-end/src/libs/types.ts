export type Message = {
    role: string;
    content: string;
};

export enum ChatState {
    HOLDING = 0,
    WAITING_CREATE = 1,
    USER_TURN = 2,
    BOT_TURN = 3,
    END = 4,
}

export type Question = {
    content: string;
    answer: string;
};

export type QuestionLabelPercent = {
    content_label: string;
    percent: string;
};

export type QuestionLabelEnglish = {
    topic: QuestionLabelPercent[];
    grammar: QuestionLabelPercent[];
    capacity: QuestionLabelPercent[];
    vocabulary: QuestionLabelPercent[];
    question_type: QuestionLabelPercent[];
    cognitive_level: QuestionLabelPercent[];
    difficult_level: QuestionLabelPercent[];
    complexity: QuestionLabelPercent[];
};
