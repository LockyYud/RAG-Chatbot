'use client';

import { useLocalStorage } from '@/lib/hooks/use-local-storage';
import { useEffect, useState } from 'react';
import { ChatList } from './chat-list';
import { ChatState, Message } from '@/lib/types';
import { ChatPanel } from './chat-panel';
import { useScrollAnchor } from '@/lib/hooks/use-scroll-anchor';
import { cn } from '@/lib/utils';
import { EmptyScreen } from './empty-screen';
import { getMessage, sendMessage } from '@/api/chat-api';
import { useChatContext } from '@/lib/context/chat-context';
import { StartBox } from './start-box';

interface InforConversation {
    grade: number;
    messages: Array<Message>;
    topic: string;
}

export function Chat() {
    const {
        state,
        setState,
        setId: setConversationId,
        id: conversationId,
    } = useChatContext();
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const [InforConversation, setInforConversation] =
        useState<InforConversation>({
            grade: 0,
            messages: [],
            topic: '',
        });

    const [listMessage, setListMessage] = useState<Message[]>([]);
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const [_, setNewChatId] = useLocalStorage('newChatId', conversationId);
    useEffect(() => {
        const getData = async () => {
            await getMessage(conversationId).then((res) => {
                setInforConversation(res);
                setListMessage(res.messages);
                setState(ChatState.USER_TURN);
            });
        };
        if (state == ChatState.WAITING_CREATE && conversationId) {
            getData();
        }
    }, [state, conversationId, setState]);

    useEffect(() => {
        if (
            conversationId &&
            listMessage.at(-1) &&
            listMessage.at(-1)?.role == 'user'
        ) {
            setState(ChatState.BOT_TURN);
            sendMessage(conversationId, listMessage.at(-1)).then((res) => {
                setState(ChatState.USER_TURN);
                setListMessage((prev) => [...prev, res]);
            });
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [conversationId, listMessage]);

    const { messagesRef, scrollRef, isAtBottom, scrollToBottom } =
        useScrollAnchor();
    useEffect(() => {
        setNewChatId(conversationId);
    }, [conversationId, setNewChatId]);
    return (
        <div
            className="bg-background group w-full overflow-auto pl-0 peer-[[data-state=open]]:lg:pl-[250px] peer-[[data-state=open]]:xl:pl-[300px]"
            ref={scrollRef}
        >
            <div className="mx-auto justify-center flex">
                <div
                    className={cn(
                        'pb-[200px] pt-4 md:pt-10 md:max-w-2xl w-full',
                    )}
                    ref={messagesRef}
                >
                    {listMessage?.length ? (
                        <ChatList
                            messages={listMessage}
                            setMessages={setListMessage}
                        />
                    ) : (
                        <div className="flex flex-col justify-center gap-6">
                            <EmptyScreen />
                            <StartBox
                                id={conversationId}
                                setId={setConversationId}
                                className="w-[400px] bg-slate-100 z-1 mx-auto rounded-3xl"
                            />
                        </div>
                    )}
                </div>
            </div>
            <ChatPanel
                setListMessage={setListMessage}
                isAtBottom={isAtBottom}
                scrollToBottom={scrollToBottom}
            />
        </div>
    );
}
