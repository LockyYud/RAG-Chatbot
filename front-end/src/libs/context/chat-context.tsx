"use client";
import { ChatState } from "@/libs/types";
import { UploadFile } from "antd";
import {
    createContext,
    useContext,
    useState,
    ReactNode,
    Dispatch,
    SetStateAction,
} from "react";

interface ChatContextInterface {
    state: ChatState;
    setState: Dispatch<SetStateAction<ChatState>>;
    id: string;
    setId: Dispatch<SetStateAction<string>>;
    fileList: UploadFile[];
    setFileList: Dispatch<SetStateAction<UploadFile[]>>;
}

const ChatContext = createContext<ChatContextInterface | undefined>(undefined);

export const useChatContext = () => {
    const context = useContext(ChatContext);
    if (context === undefined) {
        throw new Error("useChatContext must be used within a ChatProvider");
    }
    return context;
};

export default ChatContext;

interface ChatProviderProps {
    children: ReactNode;
}

export const ChatProvider: React.FC<ChatProviderProps> = ({ children }) => {
    const [state, setState] = useState<ChatState>(ChatState.HOLDING);
    const [id, setId] = useState<string>("");
    const [fileList, setFileList] = useState<UploadFile[]>([]);
    return (
        <ChatContext.Provider
            value={{ state, setState, id, setId, fileList, setFileList }}
        >
            {children}
        </ChatContext.Provider>
    );
};
