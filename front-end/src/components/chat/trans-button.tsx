'use client';
import { translateEn2Vn } from '@/api/chat-api';
import { TranslationOutlined } from '@ant-design/icons';
import { Button, Tooltip } from 'antd';
import React, { useState } from 'react';
import { cn } from '@/libs/utils';
import { useChatContext } from '@/libs/context/chat-context';
import { ChatState } from '@/libs/types';
interface TranslateProps extends React.ComponentProps<'div'> {
    content: string;
}

export const TranslationButton: React.FC<TranslateProps> = ({
    content,
    className,
}) => {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const { state, setState } = useChatContext();
    const [Trans, setTrans] = useState('');
    return (
        <div
            className={cn(
                'flex flex-row gap-2 mt-2 font-sans text-gray-500',
                className,
            )}
        >
            <Tooltip title="Not support" placement="bottom" color="#2db7f5">
                <Button
                    style={{ borderColor: 'gray' }}
                    className="hover:bg-slate-400"
                    disabled
                    onClick={() => {
                        translateEn2Vn(content).then((res) => {
                            setTrans(res);
                            setState(ChatState.BOT_TURN);
                        });
                    }}
                    icon={<TranslationOutlined style={{ color: 'gray' }} />}
                    size="small"
                />
            </Tooltip>
            <span>{Trans}</span>
        </div>
    );
};
