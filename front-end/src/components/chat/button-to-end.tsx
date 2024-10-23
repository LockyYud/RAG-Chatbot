import React from "react";
import { Button } from "antd";

interface ScrollToBottomButtonProps {
    listRef: React.RefObject<HTMLDivElement>;
}

const ScrollToBottomButton: React.FC<ScrollToBottomButtonProps> = ({
    listRef,
}) => {
    const scrollToBottom = () => {
        if (listRef.current) {
            listRef.current.scrollTop = listRef.current.scrollHeight;
        }
    };

    return (
        <Button type="primary" onClick={scrollToBottom}>
            Scroll to Bottom
        </Button>
    );
};

export default ScrollToBottomButton;
