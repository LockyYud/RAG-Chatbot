import MarkdownRenderer from "@/components/MarkdownRender/markdown";

export default function Page() {
    const text = `## Câu trả lời: Thế chiến thứ 2 diễn ra từ năm 1939 đến năm 1945.\n\n**Giải thích:**\nThông tin trong đoạn văn 1 chỉ rõ rằng Chiến tranh thế giới thứ hai bắt đầu vào \"rạng sáng 1 – 9 – 1939\" khi quân đội Đức xâm lược Ba Lan và kết thúc vào năm 1945 khi các cường quốc đã tham gia các hội nghị như Hội nghị Ianta và Pốtxđam để thiết lập trật tự thế giới mới sau chiến tranh.\n\n### Lý do tại sao các lựa chọn còn lại là sai:\n- Nếu có lựa chọn nào nói rằng Thế chiến thứ 2 diễn ra vào những năm khác (biệt lập với 1939-1945), thì những lựa chọn đó sẽ sai vì không thể hiện đúng khoảng thời gian chính xác như đã đề cập trong đoạn 1. \n- Nếu có lựa chọn nói rằng Thế chiến thứ 2 kéo dài trước hoặc sau khoảng thời gian 1939-1945, cũng sẽ không đúng vì sẽ không phù hợp với các sự kiện lịch sử đã được xác nhận trong các tài liệu."
`;
    return (
        <div>
            <MarkdownRenderer markdownText={text} />
        </div>
    );
}
