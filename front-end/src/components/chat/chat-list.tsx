'use client';
import { ChatState, Message } from '@/libs/types';
import { cn } from '@/libs/utils';
import { Divider, notification, Tooltip } from 'antd';
import { BotMessage, Suggestion, UserMessage } from './message';
import { Dispatch, SetStateAction, useEffect, useState } from 'react';
import { suggestMessage } from '@/api/chat-api';
import { useChatContext } from '@/libs/context/chat-context';
import { TranslationButton } from './trans-button';
import { FloatButton } from 'antd';
import {
    FrownTwoTone,
    LoadingOutlined,
    SignatureOutlined,
} from '@ant-design/icons';
// eslint-disable-next-line @typescript-eslint/no-unused-vars
import { AxiosError } from 'axios';
// import { AxiosError } from 'axios';
export interface ChatList {
    messages: Message[];
    setMessages: Dispatch<SetStateAction<Message[]>>;
}
export function ChatList({ messages, setMessages }: ChatList) {
    const { id: conversationId, state } = useChatContext();
    const [suggestion, setSuggestion] = useState<Message[]>([]);
    const [sugState, setSugState] = useState<boolean>(false);
    useEffect(() => {
        if (state == ChatState.BOT_TURN) {
            setSuggestion(() => []);
        }
    }, [state]);
    if (!messages.length) return null;
    const [api, contextHolder] = notification.useNotification();
    const openNotification = () => {
        api.open({
            message: `Ops, sorry`,
            description: "Right now I can't think of any suggestions for you",
            icon: <FrownTwoTone />,
            duration: 2,
        });
    };
    return (
        <div className="flex flex-col gap-6">
            {contextHolder}
            {messages.map((message, index) => (
                <div
                    key={index}
                    className={cn(
                        'flex flex-col justify-center',
                        message.role === 'user' ? 'ml-auto' : '',
                    )}
                >
                    {(message.role === 'user' && (
                        <UserMessage>{message.content}</UserMessage>
                    )) || <BotMessage>{message.content}</BotMessage>}
                    <TranslationButton
                        content={message.content}
                        className={cn(
                            message.role === 'user' ? 'float-right' : '',
                            'mr-0',
                        )}
                    />
                </div>
            ))}
            {state == ChatState.BOT_TURN && (
                <div>
                    <Divider className="my-4" />
                    <BotMessage>waiting ...</BotMessage>
                </div>
            )}
            <Suggestion
                suggestMessage={suggestion}
                setSuggestion={setSuggestion}
                setListMessage={setMessages}
            />
            {sugState && (
                <div>
                    <LoadingOutlined size={12} />
                </div>
            )}
            <Tooltip title={'Suggest Answer'}>
                <FloatButton
                    onClick={async () => {
                        setSugState(true);
                        try {
                            await suggestMessage(conversationId).then((res: SetStateAction<Message[]>) => {
                                setSuggestion(res);
                                setSugState(false);
                            });
                        // eslint-disable-next-line @typescript-eslint/no-unused-vars
                        } catch (error: AxiosError | unknown) {
                            openNotification();
                            setSugState(false);
                        }
                    }}
                    icon={<SignatureOutlined />}
                />
            </Tooltip>
        </div>
    );
}
