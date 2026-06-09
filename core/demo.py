"""core/demo.py — Demo scenarios and scripts for Polyglot Live."""

SCENARIOS = {
    1: {
        "name": "Scenario 1: Customer Support (Order Status)",
        "turns": [
            {"lang": "en", "text": "Hi, I need to check the status of my order. The order ID is 4421."},
            {"lang": "en", "text": "Yes, the email on the account is rahul@example.com."},
            {"lang": "hi", "text": "Theek hai, lekin delivery kal tak ho jaayegi kya?"},
            {"lang": "hi", "text": "Aur agar nahi hua toh refund mil sakta hai?"},
            {"lang": "en", "text": "Actually let's switch back — can you email me the tracking link?"}
        ]
    },
    2: {
        "name": "Scenario 2: Travel Planning (Hotel Booking)",
        "turns": [
            {"lang": "es", "text": "Hola, quiero reservar un hotel en Bangalore para el próximo fin de semana."},
            {"lang": "es", "text": "Para dos personas, presupuesto de 5000 rupias por noche."},
            {"lang": "en", "text": "Sorry, my Spanish is rusty. Can we continue in English? Tell me again about the second option."},
            {"lang": "en", "text": "Book it. Confirm the dates please."}
        ]
    },
    4: {
        "name": "Scenario 4: Rapid Switching (Weather Info)",
        "turns": [
            {"lang": "en", "text": "What's the weather in Mumbai today?"},
            {"lang": "hi", "text": "Aur Delhi mein?"},
            {"lang": "es", "text": "¿Y en Chennai?"},
            {"lang": "en", "text": "Compare all three for me."}
        ]
    }
}
