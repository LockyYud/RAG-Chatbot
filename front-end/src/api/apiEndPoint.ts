import { LessonPlan } from '@/app/lesson-plan/type';
import request from './request';
import { Question } from '@/lib/types';

export const createLessonPlanRequest = async (data: LessonPlan) => {
    return request.post('ai/create-lesson-plan', data);
};

export const labelingQuestionRequest = async (
    question: Question,
    subject: string,
) => {
    return request.post(`ai/label-question?subject=${subject}`, question);
};
