import React, { useState, useEffect } from "react";
import axios from "axios";
import Message from "./message";
import "./chatbot.css";

const Chatbot = () => {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [userName, setUserName] = useState("");
  const [isUserNameSet, setIsUserNameSet] = useState(false);

  useEffect(() => {
    if (isUserNameSet) {
      setMessages([
        ...messages,
        {
          sender: "bot",
          text: 'MoneyBot: Hey, Let\'s chat! (type "quit" to exit). Also, when you start bargaining, give digits.',
        },
      ]);
    }
  }, [isUserNameSet]);

  const sendMessage = async () => {
    if (input.trim() !== "") {
      if (!isUserNameSet) {
        setUserName(input);
        setIsUserNameSet(true);
        setInput("");
      } else {
        const newMessages = [
          ...messages,
          { sender: "user", text: `${userName}: ${input}` },
        ];
        setMessages(newMessages);
        const response = await axios.post(
          "https://chatbot-8ltn.onrender.com/predict",
          {
            message: input,
          }
        );
        setMessages([
          ...newMessages,
          { sender: "bot", text: response.data.response },
        ]);
        setInput("");
      }
    }
  };

  return (
    <div className="chatbot-container">
      <div className="text">Your Financial Digital Assistant</div>
      <div className="chat-window">
        <div className="messages">
          {messages.map((msg, index) => (
            <Message
              key={index}
              sender={msg.sender}
              text={msg.text}
              isUser={msg.sender === "user"}
            />
          ))}
        </div>
        <div className="input-container">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && sendMessage()}
            placeholder={
              isUserNameSet ? "Type your message..." : "Enter Your Name"
            }
          />
          <button onClick={sendMessage} className="send">
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;
