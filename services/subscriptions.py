from google.cloud import pubsub


def topics():
    pubsub_client = pubsub.Client()
    return pubsub_client.list_topics()


def create_subscription(topic_name, subscription_name):
    """Create a new pull subscription on the given topic."""
    pubsub_client = pubsub.Client()
    topic = pubsub_client.topic(topic_name)

    subscription = topic.subscription(subscription_name)
    subscription.create()

    print('Subscription {} created on topic {}.'.format(
        subscription.name, topic.name))
    return


def list_subscriptions(topic_name):
    """Lists all subscriptions for a given topic."""
    pubsub_client = pubsub.Client()
    topic = pubsub_client.topic(topic_name)
    return topic.list_subscriptions()


def delete_subscription(topic_name, subscription_name):
    """Deletes an existing Pub/Sub topic."""
    pubsub_client = pubsub.Client()
    topic = pubsub_client.topic(topic_name)
    subscription = topic.subscription(subscription_name)

    subscription.delete()

    print('Subscription {} deleted on topic {}.'.format(
        subscription.name, topic.name))
    return


def receive_message(topic_name, subscription_name):
    """Receives a message from a pull subscription."""
    pubsub_client = pubsub.Client()
    topic = pubsub_client.topic(topic_name)
    subscription = topic.subscription(subscription_name)

    # Change return_immediately=False to block until messages are
    # received.
    results = subscription.pull(return_immediately=True)
    print('Received {} messages.'.format(len(results)))
    for ack_id, message in results:
        print('* {}: {}, {}'.format(message.message_id, message.data,
                                    message.attributes))

    # Acknowledge received messages. If you do not acknowledge, Pub/Sub will
    # redeliver the message.
    if results:
        subscription.acknowledge([ack_id for ack_id, message in results])
