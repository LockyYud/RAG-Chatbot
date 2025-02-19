import { useChatContext } from "@/libs/context/chat-context";
import { List, Button, UploadFile, Tooltip } from "antd";
import { CloseOutlined } from "@ant-design/icons";
import React from "react";
export default function ListUploadFile() {
    const { fileList, setFileList } = useChatContext(); // Assuming you can update the fileList

    // Function to remove a file from the list
    const handleRemove = (file: UploadFile) => {
        const newFileList = fileList.filter(
            (item: UploadFile) => item.uid !== file.uid
        );
        setFileList(newFileList); // Update fileList after removal
    };

    return (
        <div style={{ marginTop: 16 }}>
            {fileList && fileList.length > 0 ? (
                <List
                    split={false}
                    dataSource={fileList}
                    renderItem={(file: UploadFile) => (
                        <List.Item key={file.uid} style={styles.listItem}>
                            <Tooltip title={file.name}>
                                <div style={styles.fileItem}>
                                    <span style={styles.fileName}>
                                        {file.name}
                                    </span>
                                    <Button
                                        type="text"
                                        icon={<CloseOutlined />}
                                        onClick={() => handleRemove(file)}
                                        style={styles.removeButton}
                                    />
                                </div>
                            </Tooltip>
                        </List.Item>
                    )}
                    style={styles.listContainer as React.CSSProperties}
                />
            ) : null}
        </div>
    );
}

// Inline CSS for the file item layout
const styles = {
    listContainer: {
        display: "flex", // Flexbox layout for the list
        flexDirection: "row" as const, // Align items horizontally
        overflowX: "auto", // Add horizontal scrolling if necessary
    },
    listItem: {
        display: "inline-block", // Keep each item inline
        marginRight: "6px", // Space between items
    },
    fileItem: {
        backgroundColor: "#f0f0f0",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        border: "1px solid #d9d9d9",
        borderRadius: "10px",
        padding: "3px 0px 3px 3px",
        width: "100px",
        boxSizing: "border-box" as const,
    },
    fileName: {
        padding: "0 8px",
        fontSize: "14px",
        overflow: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
    },
    removeButton: {
        border: "none",
        background: "transparent",
        cursor: "pointer",
    },
};
