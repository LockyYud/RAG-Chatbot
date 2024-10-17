'use client';
import { deleteConversation } from '@/api/chat-api';
import { Chat } from '@/components/chat/chat';
import { useChatContext } from '@/libs/context/chat-context';
import FormOutlined from '@ant-design/icons/lib/icons/FormOutlined';
import { Layout, Tooltip } from 'antd';
import { Content, Header } from 'antd/es/layout/layout';
import { useEffect } from 'react';

export default function Page() {
    const { id } = useChatContext();
    useEffect(() => {
        const handleBeforeUnload = async (event: BeforeUnloadEvent) => {
            await deleteConversation(id);
        };

        // Attach the event listener
        window.addEventListener('beforeunload', handleBeforeUnload);

        // Cleanup the event listener
        return () => {
            window.removeEventListener('beforeunload', handleBeforeUnload);
        };
    }, [id]);
    return (
        <div>
            <Layout>
                <Header
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        padding: 12,
                        gap: 6,
                        height: 56,
                        position: 'sticky',
                        top: 0,
                        zIndex: 1,
                        width: '100%',
                    }}
                    className="bg-background"
                >
                    <Tooltip
                        title="New chat"
                        placement="bottomLeft"
                        className="cursor-pointer mr-2 font-sans font-medium"
                    >
                        <FormOutlined
                            className="text-lg font-bold"
                            onClick={() => {
                                window.location.reload();
                            }}
                        />
                    </Tooltip>
                    <span className="font-sans text font-bold text-lg text-gray-500 ">
                        Sao Khue ChatBot
                    </span>
                </Header>
                <Content>
                    <Chat></Chat>
                </Content>
            </Layout>
        </div>
    );
}
