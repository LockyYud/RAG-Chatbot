'use client';
import { createConversation } from '@/api/chat-api';
import { useChatContext } from '@/libs/context/chat-context';
import { ChatState } from '@/libs/types';
import { cn } from '@/libs/utils';
import { Button, Card, Form } from 'antd';
import { Select } from 'antd';
import React from 'react';
interface BoxProps extends React.ComponentProps<'div'> {
    id: string;
    setId: (value: string) => void;
}

export function StartBox({ setId, className }: BoxProps) {
    const [topic, setTopic] = React.useState();
    const [grade, setGrade] = React.useState();
    const { state, setState } = useChatContext();
    return (
        <Card
            className={cn(
                'flex flex-col h-fit justify-center gap-6 text-center text-slate-500 font-sans',
                className,
            )}
        >
            <span className="text-center w-full text-xl font-semibold">
                Choose topic and grade of conversation
            </span>
            <Form
                onFinish={async () => {
                    if (topic && grade) {
                        setState(ChatState.WAITING_CREATE);
                        createConversation(topic, grade).then((res) => {
                            setId(res);
                        });
                    }
                }}
            >
                <div className="w-full grid grid-cols-3 gap-4 text-left pb-4 pt-4 font-medium">
                    <Form.Item required>
                        <Select
                            aria-required={true}
                            className="col-span-1"
                            showSearch
                            placeholder="Grade"
                            filterOption={(input, option) =>
                                (option?.label ?? '')
                                    .toLowerCase()
                                    .includes(input.toLowerCase())
                            }
                            onChange={(e) => {
                                setGrade(e);
                            }}
                            options={[
                                { value: '5', label: '5' },
                                { value: '6', label: '6' },
                                { value: '7', label: '7' },
                                { value: '8', label: '8' },
                                { value: '9', label: '9' },
                                { value: '10', label: '10' },
                                { value: '11', label: '11' },
                                { value: '12', label: '12' },
                            ]}
                        />
                    </Form.Item>
                    <Form.Item className="col-span-2">
                        <Select
                            showSearch
                            placeholder="Topic"
                            filterOption={(input, option) =>
                                (option?.label ?? '')
                                    .toLowerCase()
                                    .includes(input.toLowerCase())
                            }
                            onChange={(e) => {
                                setTopic(e);
                            }}
                            options={[
                                {
                                    value: 'Address',
                                    label: 'What’s your address?',
                                },
                                {
                                    value: 'Holiday',
                                    label: 'Where did you go on holiday?',
                                },
                                {
                                    value: 'Party',
                                    label: 'Did you go to the party?',
                                },
                                {
                                    value: 'Weekend',
                                    label: 'Where will you be this weekend?',
                                },
                                {
                                    value: 'Learning',
                                    label: 'How do you learn English?',
                                },
                                {
                                    value: 'Reading',
                                    label: 'What are you reading?',
                                },
                                { value: 'Family', label: 'Family life' },
                                {
                                    value: 'Environment',
                                    label: 'Humans and the environment',
                                },
                                { value: 'Music', label: 'Music' },
                                {
                                    value: 'Community',
                                    label: 'For a better community',
                                },
                            ]}
                        />
                    </Form.Item>
                </div>
                <Form.Item>
                    <Button
                        htmlType="submit"
                        loading={state == ChatState.WAITING_CREATE}
                        disabled={state == ChatState.WAITING_CREATE}
                        iconPosition="end"
                        className="w-full rounded-3xl h-16 font-sans font-semibold text-2xl bg-slate-300 primary text-slate-500 border-none hover:bg-slate-200"
                    >
                        Start conversation
                    </Button>
                </Form.Item>
            </Form>
        </Card>
    );
}
