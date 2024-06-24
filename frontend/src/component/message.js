// Message.js
import React from "react";
import "./message.css";

const Message = ({ sender, text, isUser }) => {
  const [userName, message] = text.split(": ", 2);
  return (
    <div className={`message ${isUser ? "user-message" : "bot-message"}`}>
      <span className="message-username">{userName}:</span> {message}
    </div>
  );
};

export default Message;
